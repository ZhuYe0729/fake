#!/usr/bin/env python3
"""Run selected-8 hetero quality evals with one process per GPU."""

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
DEFAULT_OUTPUT_DIR = BASELINE_ROOT / "quality/selected_8_scenarios"
DEFAULT_METHODS = ("hetero_strategy_a", "hetero_strategy_b", "hetero_strategy_c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
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
            str(SCRIPT_DIR / "eval_selected8_quality_vllm.py"),
            "--methods",
            method,
            "--output-dir",
            str(job_dir),
            "--batch-size",
            str(args.batch_size),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        log_path = job_dir / "stdout.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(f"launch {method} quality on GPU {gpu}", flush=True)
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
            has_result = (_job_dir / "selected8_vllm_quality.csv").exists()
            failures.append(
                {
                    "method": method,
                    "gpu": gpu,
                    "returncode": returncode,
                    "log": str(log_path),
                    "result_written": has_result,
                }
            )
            status = "result written" if has_result else "no result"
            print(f"FAILED {method} quality on GPU {gpu}: {log_path} ({status})", flush=True)
        else:
            print(f"done {method} quality on GPU {gpu}", flush=True)

    rows = []
    for method in methods:
        rows.extend(read_csv(args.output_dir / "jobs" / method / "selected8_vllm_quality.csv"))
    write_csv(args.output_dir / "selected8_vllm_quality.csv", rows)
    write_json(
        args.output_dir / "selected8_vllm_quality_parallel_metadata.json",
        {"methods": methods, "gpus": gpus[: len(methods)], "failures": failures},
    )
    hard_failures = [failure for failure in failures if not failure["result_written"]]
    if hard_failures:
        raise SystemExit(f"{len(hard_failures)} quality jobs failed without results")


def split_csv(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
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
