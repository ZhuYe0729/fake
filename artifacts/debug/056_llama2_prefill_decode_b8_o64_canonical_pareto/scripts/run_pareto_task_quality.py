#!/usr/bin/env python3
"""Run resumable real vLLM PMPD task shards for selected canonical policies."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
EXP = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "llama2_7b_chat"))
RUNNER = Path(os.environ.get("COSPAQ_TASK_RUNNER", REPO / "artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/scripts/run_task_quality_shard.sh"))
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}


def complete(path: Path, expected: int) -> bool:
    return path.exists() and sum(1 for _ in path.open(encoding="utf-8")) == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--policies", default="b8o64000,b8o64001,b8o64002,b8o64003,b8o64004,b8o64005,b8o64006,b8o64007,b8o64008,b8o64009")
    parser.add_argument("--shard-size", type=int, default=360)
    parser.add_argument("--iwslt-shard-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75,
                        help="Leave headroom for delayed CUDA-context teardown between fresh shards.")
    args = parser.parse_args()
    policies = tuple(item.strip() for item in args.policies.split(",") if item.strip())
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    base = EXP / "task_quality"
    tasks = [(policy, dataset, begin, min(begin + (args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size), size))
             for policy in policies for dataset, size in DATASETS.items()
             for begin in range(0, size, args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size)]
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in gpus: gpu_queue.put(gpu)

    def run(task: tuple[str, str, int, int]) -> tuple[str, str, int, int]:
        policy, dataset, begin, end = task
        checkpoint = EXP / "speed/runs" / policy / "checkpoint"
        if not (checkpoint / "model.safetensors").exists():
            raise FileNotFoundError(checkpoint)
        label = f"ours_{policy}_prefill_decode"
        shard = base / "shards" / policy / dataset / f"shard_{begin:04d}_{end:04d}"
        output = shard / dataset / f"{label}-fp16.jsonl"
        if complete(output, end - begin): return task
        gpu = gpu_queue.get()
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": gpu, "CHECKPOINT": str(checkpoint), "DATASET": dataset,
                    "QUESTION_BEGIN": str(begin), "QUESTION_END": str(end), "OUT_DIR": str(shard), "LABEL": label,
                    "BATCH_SIZE": str(args.batch_size), "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization)})
        log = base / "logs" / f"{policy}_{dataset}_{begin:04d}_{end:04d}_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open("w") as handle:
                subprocess.run(["bash", str(RUNNER)], cwd=REPO, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
        finally:
            gpu_queue.put(gpu)
        return task

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, task) for task in tasks]):
            policy, dataset, begin, end = future.result()
            print(f"completed {policy}/{dataset} [{begin},{end})", flush=True)


if __name__ == "__main__":
    main()
