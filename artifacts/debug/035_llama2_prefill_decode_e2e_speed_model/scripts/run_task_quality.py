#!/usr/bin/env python3
"""Run representative prefill-decode PMPD quality shards on a GPU pool."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SOURCE = REPO / "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver"
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
ALL_POINTS = tuple(range(12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--shard-size", type=int, default=360)
    parser.add_argument("--iwslt-shard-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-name", default="task_quality_continuous")
    parser.add_argument("--points", default=",".join(map(str, ALL_POINTS)))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    return parser.parse_args()


def checkpoint(point: int) -> Path:
    name = f"point_{point:03d}"
    candidates = (ROOT / "checkpoints" / name,
                  SOURCE / "validation/prefill_decode/checkpoints" / name)
    for candidate in candidates:
        if (candidate / "model.safetensors").exists() and (candidate / "tokenizer_config.json").exists():
            return candidate
    raise FileNotFoundError(f"missing complete checkpoint for point {point}: {candidates}")


def complete(path: Path, expected: int) -> bool:
    return path.exists() and sum(1 for _ in path.open(encoding="utf-8")) == expected


def main() -> None:
    args = parse_args()
    quality_root = ROOT / args.output_name
    points = tuple(int(item) for item in args.points.split(",") if item.strip())
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    if not points or any(point not in ALL_POINTS for point in points):
        raise ValueError(f"--points must be drawn from {ALL_POINTS}")
    if not datasets or any(dataset not in DATASETS for dataset in datasets):
        raise ValueError(f"--datasets must be drawn from {tuple(DATASETS)}")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    tasks = [(point, dataset, begin, min(begin + (args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size), size))
             for point in points for dataset, size in DATASETS.items() if dataset in datasets
             for begin in range(0, size, args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size)]
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def run(task: tuple[int, str, int, int]) -> tuple[int, str, int, int]:
        point, dataset, begin, end = task
        label = f"ours_point_{point}_prefill_decode"
        shard = quality_root / "shards" / f"point_{point}" / dataset / f"shard_{begin:04d}_{end:04d}"
        output = shard / dataset / f"{label}-fp16.jsonl"
        if complete(output, end - begin):
            return task
        gpu = gpu_queue.get()
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": gpu, "CHECKPOINT": str(checkpoint(point)), "DATASET": dataset,
                    "QUESTION_BEGIN": str(begin), "QUESTION_END": str(end), "OUT_DIR": str(shard),
                    "LABEL": label, "BATCH_SIZE": str(args.batch_size), "GPU_MEMORY_UTILIZATION": "0.85"})
        log = quality_root / "logs" / f"point_{point}_{dataset}_{begin:04d}_{end:04d}_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open("w") as handle:
                subprocess.run(["bash", str(ROOT / "scripts/run_task_quality_shard.sh")], cwd=REPO, env=env,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
        finally:
            gpu_queue.put(gpu)
        return task

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, task) for task in tasks]):
            point, dataset, begin, end = future.result()
            print(f"completed point_{point}/{dataset} [{begin},{end})", flush=True)


if __name__ == "__main__":
    main()
