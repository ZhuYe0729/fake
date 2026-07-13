#!/usr/bin/env python3
"""Run promising-policy quality evals with one process per GPU."""

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
RETEST_ROOT = SCRIPT_DIR.parent
DEFAULT_POLICY_DIR = RETEST_ROOT / "policies/unique_policies"
DEFAULT_CHECKPOINT_DIR = RETEST_ROOT / "checkpoints"
DEFAULT_OUTPUT_DIR = RETEST_ROOT / "quality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policies", default="")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = parse_policies(args)
    gpus = split_csv(args.gpus)
    if not gpus:
        raise ValueError("at least one GPU is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending = list(policies)
    running = []
    failures = []
    completed = []
    gpu_queue = list(gpus)
    while pending or running:
        while pending and gpu_queue:
            gpu = gpu_queue.pop(0)
            policy = pending.pop(0)
            running.append(launch_job(args, policy, gpu))
        next_running = []
        for job in running:
            proc = job["proc"]
            returncode = proc.poll()
            if returncode is None:
                next_running.append(job)
                continue
            job["log_file"].close()
            gpu_queue.append(job["gpu"])
            if returncode != 0:
                has_result = (job["job_dir"] / "optimized_policy_quality.csv").exists()
                failures.append(
                    {
                        "policy_name": job["policy"],
                        "gpu": job["gpu"],
                        "returncode": returncode,
                        "log": str(job["log_path"]),
                        "result_written": has_result,
                    }
                )
                status = "result written" if has_result else "no result"
                print(f"FAILED {job['policy']} quality on GPU {job['gpu']}: {job['log_path']} ({status})", flush=True)
            else:
                completed.append(job["policy"])
                print(f"done {job['policy']} quality on GPU {job['gpu']}", flush=True)
        running = next_running
        if running:
            import time

            time.sleep(5)

    rows = []
    for policy in policies:
        rows.extend(read_csv(args.output_dir / "jobs" / policy / "optimized_policy_quality.csv"))
    write_csv(args.output_dir / "optimized_policy_quality.csv", rows)
    write_json(
        args.output_dir / "optimized_policy_quality_parallel_metadata.json",
        {
            "policies": policies,
            "gpus": gpus,
            "completed": completed,
            "failures": failures,
        },
    )
    hard_failures = [failure for failure in failures if not failure["result_written"]]
    if hard_failures:
        raise SystemExit(f"{len(hard_failures)} quality jobs failed without results")


def parse_policies(args: argparse.Namespace) -> list[str]:
    if args.policies.strip():
        return split_csv(args.policies)
    policies = []
    for path in sorted(args.policy_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        policies.append(str(payload["policy_name"]))
    return policies


def launch_job(args: argparse.Namespace, policy: str, gpu: str) -> dict[str, Any]:
    job_dir = args.output_dir / "jobs" / policy
    job_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        args.python,
        str(SCRIPT_DIR / "eval_promising_policy_quality_vllm.py"),
        "--policies",
        policy,
        "--output-dir",
        str(job_dir),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
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
    print(f"launch {policy} quality on GPU {gpu}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(RETEST_ROOT.parents[4]),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "policy": policy,
        "gpu": gpu,
        "proc": proc,
        "log_file": log_file,
        "log_path": log_path,
        "job_dir": job_dir,
    }


def split_csv(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
