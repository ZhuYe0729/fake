#!/usr/bin/env python3
"""Run selected-8 vLLM benchmark jobs with one process per GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = BASELINE_ROOT / "benchmarks/selected_8_scenarios_vllm"
DEFAULT_METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "hetero",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = split_csv(args.methods)
    gpus = split_csv(args.gpus)
    if len(gpus) < len(methods):
        raise ValueError(f"need at least {len(methods)} GPUs for methods={methods}, got {gpus}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for gpu, method in zip(gpus, methods, strict=True):
        job_dir = args.output_dir / "jobs" / method
        job_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            args.python,
            str(SCRIPT_DIR / "benchmark_selected8_vllm.py"),
            "--methods",
            method,
            "--output-dir",
            str(job_dir),
            "--warmup-iters",
            str(args.warmup_iters),
            "--iters",
            str(args.iters),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--seed",
            str(args.seed),
        ]
        if args.continue_on_error:
            cmd.append("--continue-on-error")
        log_path = job_dir / "stdout.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(f"launch {method} on GPU {gpu}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASELINE_ROOT.parents[3]),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        jobs.append((method, gpu, proc, log_file, log_path, job_dir))

    failures = []
    for method, gpu, proc, log_file, log_path, _job_dir in jobs:
        returncode = proc.wait()
        log_file.close()
        if returncode != 0:
            failures.append(
                {
                    "method": method,
                    "gpu": gpu,
                    "returncode": returncode,
                    "log": str(log_path),
                }
            )
            print(f"FAILED {method} on GPU {gpu}: {log_path}", flush=True)
        else:
            print(f"done {method} on GPU {gpu}", flush=True)

    merge_job_outputs(args.output_dir, methods)
    write_json(
        args.output_dir / "selected8_vllm_parallel_metadata.json",
        {
            "methods": methods,
            "gpus": gpus[: len(methods)],
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "failures": failures,
        },
    )
    if failures and not args.continue_on_error:
        raise SystemExit(f"{len(failures)} benchmark jobs failed")


def split_csv(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def merge_job_outputs(output_dir: Path, methods: list[str]) -> None:
    summaries: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for method in methods:
        job_dir = output_dir / "jobs" / method
        summaries.extend(read_csv(job_dir / "selected8_vllm_summary.csv"))
        iterations.extend(read_csv(job_dir / "selected8_vllm_iterations.csv"))
        meta_path = job_dir / "selected8_vllm_metadata.json"
        if meta_path.exists():
            metadata.append(json.loads(meta_path.read_text()))
    add_speedups(summaries)
    write_csv(output_dir / "selected8_vllm_summary.csv", summaries)
    write_csv(output_dir / "selected8_vllm_iterations.csv", iterations)
    write_json(output_dir / "selected8_vllm_metadata.json", {"jobs": metadata})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_speedups(rows: list[dict[str, Any]]) -> None:
    dense_by_scenario = {
        row["scenario"]: float(row["median_ms"])
        for row in rows
        if row.get("method") == "dense_bf16" and row.get("median_ms")
    }
    for row in rows:
        dense_ms = dense_by_scenario.get(row.get("scenario", ""))
        try:
            median_ms = float(row["median_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if dense_ms and median_ms:
            row["speedup_vs_dense_bf16"] = dense_ms / median_ms


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
