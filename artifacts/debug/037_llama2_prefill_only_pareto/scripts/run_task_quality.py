#!/usr/bin/env python3
"""Run representative prefill-only policies on the three PMPD generation tasks."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
DEFAULT_POINTS = (6, 9, 15)
SHARD = ROOT / "scripts/run_task_quality_shard.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--points", default=",".join(map(str, DEFAULT_POINTS)))
    parser.add_argument("--shard-size", type=int, default=360)
    parser.add_argument("--iwslt-shard-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def complete(path: Path, expected: int) -> bool:
    if not path.exists() or sum(1 for _ in path.open(encoding="utf-8")) != expected:
        return False
    try:
        for line in path.open(encoding="utf-8"):
            json.loads(line)
    except json.JSONDecodeError:
        return False
    return True


def main() -> None:
    args = parse_args()
    points = tuple(int(value) for value in args.points.split(",") if value.strip())
    if not points or any(point not in DEFAULT_POINTS for point in points):
        raise ValueError(f"--points must be a subset of {DEFAULT_POINTS}")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    tasks = []
    for point in points:
        checkpoint = ROOT / "checkpoints" / f"point_{point:03d}"
        if not (checkpoint / "model.safetensors").exists():
            raise FileNotFoundError(checkpoint)
        for dataset, size in DATASETS.items():
            step = args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size
            tasks.extend((point, dataset, begin, min(begin + step, size)) for begin in range(0, size, step))
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def run(task: tuple[int, str, int, int]) -> tuple[int, str, int, int]:
        point, dataset, begin, end = task
        label = f"ours_point_{point}_prefill_only"
        shard = ROOT / "task_quality/shards" / f"point_{point}" / dataset / f"shard_{begin:04d}_{end:04d}"
        output = shard / dataset / f"{label}-fp16.jsonl"
        if complete(output, end - begin):
            return task
        gpu = gpu_queue.get()
        env = os.environ.copy()
        env.update({"PHYSICAL_GPU": gpu, "CHECKPOINT": str(ROOT / "checkpoints" / f"point_{point:03d}"),
                    "DATASET": dataset, "QUESTION_BEGIN": str(begin), "QUESTION_END": str(end),
                    "OUT_DIR": str(shard), "LABEL": label, "BATCH_SIZE": str(args.batch_size),
                    "GPU_MEMORY_UTILIZATION": "0.75"})
        log = ROOT / "task_quality/logs" / f"point_{point}_{dataset}_{begin:04d}_{end:04d}_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            for attempt in range(args.retries):
                with log.open("w") as handle:
                    completed = subprocess.run(["bash", str(SHARD)], cwd=REPO, env=env, stdout=handle,
                                               stderr=subprocess.STDOUT).returncode == 0
                if completed:
                    return task
                if attempt + 1 < args.retries:
                    time.sleep(15)
            raise RuntimeError(f"failed after {args.retries} attempts: {task}")
        finally:
            gpu_queue.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, task) for task in tasks]):
            try:
                point, dataset, begin, end = future.result()
                print(f"completed point_{point}/{dataset} [{begin},{end})", flush=True)
            except Exception as error:
                print(f"failed {error}", flush=True)


if __name__ == "__main__":
    main()
