#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from common_pareto import DEBUG_ROOT, f, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stable E2E repeats as one fresh process per repeat.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--points", default="0,7,9")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters-per-process", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--alloc-conf", default="expandable_segments:True")
    parser.add_argument("--run-name", default=None, help="Optional subdirectory name under validation/stable_e2e_repeats.")
    parser.add_argument("--validator-script", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = [int(item) for item in args.points.split(",") if item.strip()]
    out_dir = args.output_root / "validation" / "stable_e2e_repeats"
    if args.run_name:
        out_dir = out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_csv = args.output_root / "validation" / "pareto_e2e_validation.csv"
    backup_csv = out_dir / "pareto_e2e_validation_before_stable_repeats.csv"
    if validation_csv.exists():
        shutil.copy2(validation_csv, backup_csv)

    raw_rows: list[dict[str, Any]] = []
    try:
        for point in points:
            for repeat in range(args.repeats):
                print(f"running point={point} repeat={repeat}")
                status, stdout_tail, stderr_tail = run_one(args, point)
                row = collect_row(validation_csv, point, repeat, status, stdout_tail, stderr_tail)
                raw_rows.append(row)
                write_csv(out_dir / "stable_e2e_repeats_raw.csv", raw_rows)
    finally:
        if backup_csv.exists():
            shutil.copy2(backup_csv, validation_csv)

    summary_rows = summarize(raw_rows)
    write_csv(out_dir / "stable_e2e_repeats_summary.csv", summary_rows)
    write_json(
        out_dir / "stable_e2e_repeats_metadata.json",
        {
            "points": points,
            "gpu": args.gpu,
            "repeats": args.repeats,
            "warmup_iters": args.warmup_iters,
            "iters_per_process": args.iters_per_process,
            "backup_csv": str(backup_csv),
        },
    )
    print(f"wrote {len(raw_rows)} raw rows and {len(summary_rows)} summary rows to {out_dir}")


def run_one(args: argparse.Namespace, point: int) -> tuple[str, str, str]:
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = args.alloc_conf
    env["PYTORCH_CUDA_ALLOC_CONF"] = args.alloc_conf
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or str(args.gpu)
    cmd = [
        sys.executable,
        str(args.validator_script or (args.output_root / "scripts" / "validate_pareto_e2e.py")),
        "--output-root",
        str(args.output_root),
        "--gpu",
        str(args.gpu),
        "--points",
        str(point),
        "--warmup-iters",
        str(args.warmup_iters),
        "--iters",
        str(args.iters_per_process),
    ]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    status = "ok" if proc.returncode == 0 else f"process_failed:{proc.returncode}"
    return status, tail(proc.stdout), tail(proc.stderr)


def collect_row(path: Path, point: int, repeat: int, process_status: str, stdout_tail: str, stderr_tail: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "point_index": point,
        "repeat_index": repeat,
        "process_status": process_status,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    if path.exists():
        rows = read_csv(path)
        match = next((item for item in rows if int(f(item, "point_index")) == point), None)
        if match:
            row.update(match)
    if row.get("e2e_status") != "ok" and process_status == "ok":
        row["process_status"] = "validation_error"
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    points = sorted({int(f(row, "point_index")) for row in rows})
    for point in points:
        items = [row for row in rows if int(f(row, "point_index")) == point]
        ok = [row for row in items if row.get("e2e_status") == "ok" and row.get("e2e_total_mean_ms", "") != ""]
        vals = [f(row, "e2e_total_mean_ms") for row in ok]
        prefill = [f(row, "e2e_prefill_mean_ms") for row in ok]
        decode = [f(row, "e2e_decode_avg_mean_ms") for row in ok]
        first = ok[0] if ok else items[0]
        out.append(
            {
                "point_index": point,
                "attempts": len(items),
                "ok_repeats": len(ok),
                "failed_repeats": len(items) - len(ok),
                "quality_cost": first.get("quality_cost", ""),
                "predicted_total_latency_ms": first.get("predicted_total_latency_ms", ""),
                "e2e_total_mean_ms": mean(vals) if vals else "",
                "e2e_total_median_ms": median(vals) if vals else "",
                "e2e_total_min_ms": min(vals) if vals else "",
                "e2e_total_max_ms": max(vals) if vals else "",
                "e2e_total_std_ms": stdev(vals) if len(vals) >= 2 else "",
                "e2e_prefill_mean_ms": mean(prefill) if prefill else "",
                "e2e_decode_avg_mean_ms": mean(decode) if decode else "",
                "backend_counts": first.get("backend_counts", ""),
            }
        )
    dense = next((f(row, "e2e_total_mean_ms") for row in out if int(f(row, "point_index")) == 0), None)
    for row in out:
        total = f(row, "e2e_total_mean_ms")
        row["e2e_speedup_vs_point0"] = dense / total if dense and total > 0 else ""
    return out


def tail(text: str, lines: int = 20) -> str:
    parts = text.strip().splitlines()
    return "\\n".join(parts[-lines:])


if __name__ == "__main__":
    main()
