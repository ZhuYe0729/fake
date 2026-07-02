#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEBUG030 = ROOT / "artifacts" / "debug" / "030_mirror_global_pareto"
DEBUG032 = ROOT / "artifacts" / "debug" / "032_mirror_speed_regression_debug"
RUNNER = DEBUG030 / "scripts" / "run_batch_speed_sweep.py"
SUMMARIZER = DEBUG030 / "scripts" / "summarize_batch_speed_sweep.py"

BATCHES = (1, 2, 4, 8, 16, 32)
LABELS = (
    "dense_default_amp",
    "uniform_dense_bf16",
    "uniform_dense_nvfp4",
    "uniform_sparse_bf16",
    "uniform_sparse_nvfp4",
    "ours_gate_up_sparse_bf16_64",
    "ours_mlp_sparse_bf16_96",
    "ours_mlp_all_attn_sparse_bf16_64",
    "ours_extreme_fastest_microbench",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected MIRROR batch sweep with each batch pinned to one GPU.")
    parser.add_argument("--output-root", type=Path, default=DEBUG032 / "corrected_batch_speed_sweep_by_batch")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--labels", nargs="+", default=list(LABELS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    assignments = [(batch, args.gpus[i % len(args.gpus)]) for i, batch in enumerate(args.batch_sizes)]
    with ThreadPoolExecutor(max_workers=min(len(args.gpus), len(args.batch_sizes))) as pool:
        futures = [pool.submit(run_batch, args, batch, gpu) for batch, gpu in assignments]
        for future in as_completed(futures):
            future.result()
    aggregate_csv = args.output_root / "corrected_batch_speed_sweep_by_batch.csv"
    aggregate(args.output_root, aggregate_csv)
    report_dir = args.output_root / "report"
    subprocess.run(
        [sys.executable, str(SUMMARIZER), "--input-csv", str(aggregate_csv), "--output-dir", str(report_dir)],
        cwd=ROOT,
        check=True,
    )
    public_report = DEBUG030 / "speedaware_frontier" / "report"
    public_report.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        src = report_dir / f"batch_speed_sweep_speedup{suffix}"
        shutil.copy2(src, public_report / f"batch_speed_sweep_speedup_corrected_by_batch{suffix}")
    shutil.copy2(aggregate_csv, public_report / "batch_speed_sweep_corrected_by_batch.csv")
    print(f"wrote {aggregate_csv}")
    print(f"wrote {report_dir / 'batch_speed_sweep_speedup.png'}")


def run_batch(args: argparse.Namespace, batch: int, gpu: int) -> None:
    logs = args.output_root / "logs" / f"batch_{batch}"
    logs.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        out_csv = args.output_root / "single" / f"batch_{batch}" / f"{label}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[skip] batch={batch} label={label}")
            continue
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{label}.log"
        cmd = [
            sys.executable,
            str(RUNNER),
            "--gpu",
            "0",
            "--batch-sizes",
            str(batch),
            "--labels",
            label,
            "--output-csv",
            str(out_csv),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--overwrite",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"[done] gpu={gpu} batch={batch} label={label}")


def aggregate(output_root: Path, out_csv: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in sorted((output_root / "single").glob("batch_*/*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("no result rows found")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
