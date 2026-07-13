#!/usr/bin/env python3
"""Run broad-grid vLLM benchmark with one method process per GPU."""

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
BROAD_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BROAD_ROOT.parents[4]
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
    parser.add_argument("--output-dir", type=Path, default=BROAD_ROOT / "results")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--gpus", default="1,2,3,4,5,6")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batches", default="")
    parser.add_argument("--input-seqs", default="")
    parser.add_argument("--output-seqs", default="")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-total-prompt-tokens", type=int, default=131072)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.pop("_CUDA_COMPAT_STATUS", None)
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
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
        env.pop("_CUDA_COMPAT_STATUS", None)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            args.python,
            str(SCRIPT_DIR / "benchmark_broad_grid_vllm.py"),
            "--method",
            method,
            "--output-dir",
            str(job_dir),
            "--warmup-iters",
            str(args.warmup_iters),
            "--iters",
            str(args.iters),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-total-prompt-tokens",
            str(args.max_total_prompt_tokens),
            "--continue-on-error",
        ]
        if args.batches:
            cmd.extend(["--batches", args.batches])
        if args.input_seqs:
            cmd.extend(["--input-seqs", args.input_seqs])
        if args.output_seqs:
            cmd.extend(["--output-seqs", args.output_seqs])
        log_path = job_dir / "stdout.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(f"launch {method} on GPU {gpu}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        jobs.append((method, gpu, proc, log_file, log_path))

    failures = []
    for method, gpu, proc, log_file, log_path in jobs:
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

    merge_outputs(args.output_dir, methods)
    write_json(
        args.output_dir / "parallel_metadata.json",
        {
            "methods": methods,
            "gpus": gpus[: len(methods)],
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_total_prompt_tokens": args.max_total_prompt_tokens,
            "failures": failures,
        },
    )
    if failures and not args.continue_on_error:
        raise SystemExit(f"{len(failures)} benchmark jobs failed")


def split_csv(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def merge_outputs(output_dir: Path, methods: list[str]) -> None:
    summaries: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    metadata = []
    for method in methods:
        job_dir = output_dir / "jobs" / method
        summaries.extend(read_csv(job_dir / "summary_long.csv"))
        iterations.extend(read_csv(job_dir / "iterations.csv"))
        meta_path = job_dir / "metadata.json"
        if meta_path.exists():
            metadata.append(json.loads(meta_path.read_text()))
    write_csv(output_dir / "summary_long.csv", summaries)
    write_csv(output_dir / "iterations.csv", iterations)
    write_json(output_dir / "metadata.json", {"jobs": metadata})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
