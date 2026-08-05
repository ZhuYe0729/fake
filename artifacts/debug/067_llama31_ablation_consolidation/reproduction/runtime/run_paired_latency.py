#!/usr/bin/env python3
"""Run alternating sync-copy and async-overlap E2E latency samples."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
CHECKPOINT = EXP / "checkpoint/point_011"
BENCHMARK = (ROOT / "artifacts/exports/vllm/ours/llama2-7b-chat/scripts"
             / "benchmark_phase_hetero.py")
VLLM_PYTHON = Path(os.environ.get(
    "VLLM_PYTHON", "/home/agent/wja/miniconda3/envs/vllm/bin/python"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--max-pre-run-memory-mib", type=int, default=512)
    return parser.parse_args()


def gpu_state(gpu: str) -> dict[str, int]:
    command = [
        "nvidia-smi", "-i", gpu,
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    last_error: subprocess.CalledProcessError | None = None
    for _ in range(3):
        try:
            fields = subprocess.check_output(
                command, text=True, stderr=subprocess.STDOUT).strip().split(",")
            return {"memory_used_mib": int(fields[0].strip()),
                    "utilization_percent": int(fields[1].strip())}
        except subprocess.CalledProcessError as error:
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"nvidia-smi failed after three attempts: {last_error}")


def run_one(gpu: str, variant: str, sync_copy: str, pair: int,
            warmup: bool, memory_limit: int) -> dict[str, object]:
    state = gpu_state(gpu)
    if state["memory_used_mib"] > memory_limit:
        raise RuntimeError(
            f"GPU {gpu} is not idle before {variant} pair {pair}: {state}")
    run_dir = EXP / "raw" / ("warmup" if warmup else "measured")
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pair_{pair:02d}_{variant}"
    result = run_dir / f"{stem}.json"
    log = run_dir / f"{stem}.log"
    command = [
        str(VLLM_PYTHON), str(BENCHMARK),
        "--checkpoint", str(CHECKPOINT),
        "--scenario", "prefill_decode",
        "--batch", "8", "--input-seq", "2048", "--output-seq", "64",
        "--output-dir", str(run_dir),
        "--gpu-memory-utilization", "0.80",
        "--warmup-iters", "0", "--iters", "1",
        "--single-phase", "main", "--single-output", str(result),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "VLLM_USE_V1": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PHASE_HETERO_TRACE": "0",
        "PHASE_HETERO_SYNC_COPY": sync_copy,
        "PHASE_HETERO_GC_EVERY_APPLY": "1",
        "PHASE_HETERO_GC_DECODE": "0",
        "PHASE_HETERO_WAIT_ONCE": "1",
        "PHASE_HETERO_RELEASE_PREFILL": "1",
    })
    started = time.time()
    with log.open("w") as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    elapsed_ms = float(json.loads(result.read_text())["elapsed_ms"])
    return {
        "pair": pair,
        "variant": variant,
        "sync_copy": int(sync_copy),
        "warmup": int(warmup),
        "e2e_latency_ms": elapsed_ms,
        "gpu": gpu,
        "pre_run_memory_mib": state["memory_used_mib"],
        "pre_run_utilization_percent": state["utilization_percent"],
        "wall_seconds": time.time() - started,
        "log": str(log.relative_to(EXP)),
    }


def write_rows(rows: list[dict[str, object]]) -> None:
    path = EXP / "raw_samples.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    lock_handle = (EXP / ".run.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another overlap ablation runner is active") from error
    if not (CHECKPOINT / "model.safetensors").exists():
        raise FileNotFoundError(
            "checkpoint missing; run scripts/materialize_checkpoint.sh first")
    sample_path = EXP / "raw_samples.csv"
    rows: list[dict[str, object]] = []
    if sample_path.exists():
        rows = list(csv.DictReader(sample_path.open()))
    completed = {(int(row["pair"]), row["variant"]) for row in rows}
    total = args.warmup_pairs + args.pairs
    for pair in range(total):
        warmup = pair < args.warmup_pairs
        # AB/BA alternation cancels slow temporal drift without parallel speed runs.
        variants = [("synchronous_copy", "1"),
                    ("asynchronous_overlap", "0")]
        if pair % 2:
            variants.reverse()
        for variant, flag in variants:
            if (pair, variant) in completed:
                continue
            row = run_one(args.gpu, variant, flag, pair, warmup,
                          args.max_pre_run_memory_mib)
            rows.append(row)
            completed.add((pair, variant))
            write_rows(rows)
            print(json.dumps(row), flush=True)
    metadata = {
        "model": "Llama3.1-8B-Instruct",
        "policy": "point_011",
        "scenario": {"batch": 8, "input_seq": 2048, "output_seq": 64},
        "gpus": sorted({str(row["gpu"]) for row in rows}),
        "warmup_pairs": args.warmup_pairs,
        "measured_pairs": args.pairs,
        "order": "alternating AB/BA",
        "trace_enabled": False,
        "varying_environment_variable": "PHASE_HETERO_SYNC_COPY",
    }
    (EXP / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
