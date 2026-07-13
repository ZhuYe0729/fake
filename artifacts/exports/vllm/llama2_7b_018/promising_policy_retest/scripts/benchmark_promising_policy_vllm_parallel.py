#!/usr/bin/env python3
"""Run optimized hetero promising-scenario benchmarks with one job per GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RETEST_ROOT = SCRIPT_DIR.parent
DEFAULT_POLICY_CSV = RETEST_ROOT / "policies/scenario_policies.csv"
DEFAULT_CHECKPOINT_ROOT = RETEST_ROOT / "checkpoints"
DEFAULT_OUTPUT_DIR = RETEST_ROOT / "benchmarks/optimized_hetero_vllm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-csv", type=Path, default=DEFAULT_POLICY_CSV)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method-name", default="optimized_hetero")
    parser.add_argument("--output-prefix", default="optimized_hetero")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "1,2,3,4,5,6"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--scenarios", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = filter_scenarios(read_scenario_names(args.policy_csv), args.scenarios)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("no GPUs specified")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = args.output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    pending = list(scenarios)
    running: dict[subprocess.Popen, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []

    while pending or running:
        while pending and len(running) < len(gpus):
            scenario = pending.pop(0)
            gpu = first_free_gpu(gpus, running)
            job_dir = jobs_dir / scenario
            job_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.python,
                str(SCRIPT_DIR / "benchmark_promising_policy_vllm.py"),
                "--policy-csv",
                str(args.policy_csv),
                "--checkpoint-root",
                str(args.checkpoint_root),
                "--output-dir",
                str(job_dir),
                "--method-name",
                args.method_name,
                "--output-prefix",
                args.output_prefix,
                "--scenarios",
                scenario,
                "--warmup-iters",
                str(args.warmup_iters),
                "--iters",
                str(args.iters),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--continue-on-error",
            ]
            env = os.environ.copy()
            env.pop("_CUDA_COMPAT_STATUS", None)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            stdout = (job_dir / "stdout.log").open("w")
            print(f"launch {scenario} on GPU {gpu}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(RETEST_ROOT.parents[4]), env=env, stdout=stdout, stderr=subprocess.STDOUT)
            running[proc] = {"scenario": scenario, "gpu": gpu, "stdout": stdout, "start_time": time.time()}

        time.sleep(2)
        for proc, meta in list(running.items()):
            if proc.poll() is None:
                continue
            meta["stdout"].close()
            meta["returncode"] = proc.returncode
            meta["elapsed_sec"] = time.time() - float(meta["start_time"])
            completed.append(meta)
            del running[proc]
            print(f"done {meta['scenario']} on GPU {meta['gpu']} rc={proc.returncode}", flush=True)

    summary_rows = merge_csvs(jobs_dir, f"{args.output_prefix}_summary.csv")
    iteration_rows = merge_csvs(jobs_dir, f"{args.output_prefix}_iterations.csv")
    write_csv(args.output_dir / f"{args.output_prefix}_summary.csv", summary_rows)
    write_csv(args.output_dir / f"{args.output_prefix}_iterations.csv", iteration_rows)
    write_json(
        args.output_dir / f"{args.output_prefix}_parallel_metadata.json",
        {
            "gpus": gpus,
            "scenarios": scenarios,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "completed": [{k: v for k, v in row.items() if k != "stdout"} for row in completed],
        },
    )


def read_scenario_names(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return [row["scenario"] for row in csv.DictReader(f)]


def filter_scenarios(names: list[str], spec: str) -> list[str]:
    if spec == "all":
        return names
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"unknown scenarios: {missing}")
    return requested


def first_free_gpu(gpus: list[str], running: dict[subprocess.Popen, dict[str, Any]]) -> str:
    used = {meta["gpu"] for meta in running.values()}
    for gpu in gpus:
        if gpu not in used:
            return gpu
    raise RuntimeError("no free GPU")


def merge_csvs(jobs_dir: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(jobs_dir.glob(f"*/{name}")):
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
