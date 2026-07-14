#!/usr/bin/env python3
"""Run continuous phase-hetero PMPD shards for intermediate policies only."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
POINTS = (34, 36, 37, 38)
SHARD = REPO / "artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/scripts/run_task_quality_shard.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--output-name", default="task_quality_intermediate")
    parser.add_argument("--shard-size", type=int, default=360)
    parser.add_argument("--iwslt-shard-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    quality_root = ROOT / args.output_name
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    tasks = [(point, dataset, begin, min(begin + (args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size), size))
             for point in POINTS for dataset, size in DATASETS.items()
             for begin in range(0, size, args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size)]
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def run(task: tuple[int, str, int, int]) -> tuple[int, str, int, int]:
        point, dataset, begin, end = task
        label = f"ours_intermediate_{point}_prefill_decode"
        shard = quality_root / "shards" / f"point_{point}" / dataset / f"shard_{begin:04d}_{end:04d}"
        output = shard / dataset / f"{label}-fp16.jsonl"
        if output.exists() and sum(1 for _ in output.open(encoding="utf-8")) == end - begin:
            return task
        checkpoint = ROOT / "checkpoints" / f"point_{point:03d}"
        if not (checkpoint / "model.safetensors").exists():
            raise FileNotFoundError(checkpoint)
        gpu = gpu_queue.get()
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": gpu, "CHECKPOINT": str(checkpoint), "DATASET": dataset,
                    "QUESTION_BEGIN": str(begin), "QUESTION_END": str(end), "OUT_DIR": str(shard),
                    "LABEL": label, "BATCH_SIZE": str(args.batch_size), "GPU_MEMORY_UTILIZATION": "0.85"})
        log = quality_root / "logs" / f"point_{point}_{dataset}_{begin:04d}_{end:04d}_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open("w") as handle:
                subprocess.run(["bash", str(SHARD)], cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        finally:
            gpu_queue.put(gpu)
        return task

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, task) for task in tasks]):
            point, dataset, begin, end = future.result()
            print(f"completed point_{point}/{dataset} [{begin},{end})", flush=True)


if __name__ == "__main__":
    main()
