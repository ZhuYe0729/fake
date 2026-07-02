#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
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
    parser = argparse.ArgumentParser(description="Run corrected MIRROR batch speed sweep with one policy per process.")
    parser.add_argument("--output-root", type=Path, default=DEBUG032 / "corrected_batch_speed_sweep")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--labels", nargs="+", default=list(LABELS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"batch_size": batch, "label": label}
        for batch in args.batch_sizes
        for label in args.labels
        if args.overwrite or not single_csv(args.output_root, batch, label).exists()
    ]
    run_jobs(args, jobs)
    aggregate_csv = args.output_root / "corrected_batch_speed_sweep.csv"
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
        dst = public_report / f"batch_speed_sweep_speedup_corrected{suffix}"
        shutil.copy2(src, dst)
    shutil.copy2(aggregate_csv, public_report / "batch_speed_sweep_corrected.csv")
    print(f"wrote {aggregate_csv}")
    print(f"wrote {report_dir / 'batch_speed_sweep_speedup.png'}")


def single_csv(output_root: Path, batch_size: int, label: str) -> Path:
    return output_root / "single" / f"batch_{batch_size}" / f"{label}.csv"


def run_jobs(args: argparse.Namespace, jobs: list[dict[str, object]]) -> None:
    if not jobs:
        return
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active: list[dict[str, object]] = []
    queue = list(jobs)
    while queue or active:
        busy_gpus = {int(item["gpu"]) for item in active}
        free_gpus = [gpu for gpu in args.gpus if gpu not in busy_gpus]
        while queue and free_gpus:
            job = queue.pop(0)
            gpu = free_gpus.pop(0)
            batch_size = int(job["batch_size"])
            label = str(job["label"])
            out_csv = single_csv(args.output_root, batch_size, label)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"batch_{batch_size}_{label}.log"
            cmd = [
                sys.executable,
                str(RUNNER),
                "--gpu",
                "0",
                "--batch-sizes",
                str(batch_size),
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
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"proc": proc, "log": log, "batch_size": batch_size, "label": label, "gpu": gpu})
            print(f"[launch] gpu={gpu} batch={batch_size} label={label}")
        time.sleep(2.0)
        next_active: list[dict[str, object]] = []
        for item in active:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.poll()
            if code is None:
                next_active.append(item)
                continue
            log = item["log"]
            assert hasattr(log, "close")
            log.close()
            batch_size = item["batch_size"]
            label = item["label"]
            if code != 0:
                raise RuntimeError(f"job failed: batch={batch_size} label={label}; see logs")
            print(f"[done] batch={batch_size} label={label}")
        active = next_active


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
