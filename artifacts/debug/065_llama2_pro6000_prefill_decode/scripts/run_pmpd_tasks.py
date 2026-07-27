#!/usr/bin/env python3
"""Export selected policies and run resumable PMPD generation shards."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

from common import CUTLASS, EXPORTER, MODEL, PMPD, RUN, runtime_env

HERE = Path(__file__).resolve().parent


def policy_path(label: str) -> Path:
    if label.startswith("uniform_"):
        return RUN / "policies/prefill_decode" / f"{label.removeprefix('uniform_')}.json"
    return RUN / "pareto/policies" / f"{label}.json"


def complete(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return len(rows) == expected and len({row["question_id"] for row in rows}) == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--iwslt-shard-size", type=int, default=100)
    args = parser.parse_args()
    selected = json.loads((RUN / "tasks/selection.json").read_text())["selected"]
    canonical_bf16 = RUN / "canonical/prepared/sparse_bf16/model.pt"
    canonical_nvfp4 = RUN / "canonical/prepared/sparse_nvfp4/model.pt"
    checkpoints = {}
    for label in selected:
        policy = policy_path(label)
        checkpoint = RUN / "tasks/checkpoints" / label
        if not (checkpoint / "phase_hetero_provenance.json").is_file():
            subprocess.run([sys.executable, str(EXPORTER), "--model-path", str(MODEL),
                            "--policy-json", str(policy), "--output-dir", str(checkpoint),
                            "--cutlass-wrapper-path", str(CUTLASS),
                            "--canonical-sparse-bf16-state", str(canonical_bf16),
                            "--canonical-sparse-nvfp4-state", str(canonical_nvfp4)],
                           check=True, env=runtime_env())
        audit = subprocess.run([sys.executable, str(HERE / "verify_checkpoint.py"),
                                "--policy", str(policy), "--checkpoint", str(checkpoint),
                                "--canonical-bf16", str(canonical_bf16),
                                "--canonical-nvfp4", str(canonical_nvfp4)],
                               check=True, text=True, capture_output=True, env=runtime_env())
        audit_path = RUN / f"tasks/audits/{label}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True); audit_path.write_text(audit.stdout)
        checkpoints[label] = checkpoint
    jobs = []
    for label in selected:
        for dataset, size in PMPD["datasets"].items():
            shard_size = args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size
            for begin in range(0, size, shard_size):
                end = min(size, begin + shard_size)
                output = RUN / f"tasks/shards/{label}/{dataset}/{begin:04d}_{end:04d}/{dataset}/{label}-fp16.jsonl"
                if not complete(output, end - begin):
                    jobs.append((label, dataset, begin, end, output))
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in (value.strip() for value in args.gpus.split(",") if value.strip()): gpu_queue.put(gpu)
    if gpu_queue.empty(): raise ValueError("no GPUs")

    def run(job):
        label, dataset, begin, end, output = job
        gpu = gpu_queue.get()
        try:
            for attempt, (batch, util) in enumerate(((4, .75), (1, .60), (1, .50))):
                output.parent.parent.parent.mkdir(parents=True, exist_ok=True)
                out_root = output.parents[1]
                env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = gpu
                command = [sys.executable, str(HERE / "run_generation_shard.py"),
                           "--dataset", dataset, "--question-begin", str(begin), "--question-end", str(end),
                           "--batch-size", str(batch), "--model-path", str(checkpoints[label]),
                           "--model-id", label, "--output-dir", str(out_root), "--phase-hetero",
                           "--max-num-batched-tokens", "15360", "--max-model-len", "4096",
                           "--gpu-memory-utilization", str(util), "--skip-metrics"]
                log = RUN / f"tasks/logs/{label}_{dataset}_{begin:04d}_{end:04d}_gpu{gpu}_try{attempt}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w") as handle:
                    result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode == 0 and complete(output, end - begin):
                    return job
            raise RuntimeError(f"task shard failed after retries: {label}/{dataset}/{begin}:{end}")
        finally:
            gpu_queue.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_queue.qsize()) as pool:
        for result in concurrent.futures.as_completed([pool.submit(run, job) for job in jobs]):
            label, dataset, begin, end, _ = result.result()
            print(f"completed {label}/{dataset}[{begin},{end})", flush=True)


if __name__ == "__main__":
    main()
