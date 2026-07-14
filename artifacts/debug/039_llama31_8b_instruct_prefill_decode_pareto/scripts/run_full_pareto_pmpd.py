#!/usr/bin/env python3
"""Evaluate one closed Pareto point with persistent PMPD vLLM shards."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIZES = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", default="point_004")
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def complete(path: Path, count: int) -> bool:
    return path.exists() and sum(1 for _ in path.open(encoding="utf-8")) == count


def main() -> None:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus or any(gpu not in {str(i) for i in range(1, 8)} for gpu in gpus):
        raise ValueError("--gpus must be a non-empty subset of 1..7")
    jobs = [(dataset, begin, min(begin + args.shard_size, size))
            for dataset, size in SIZES.items()
            for begin in range(0, size, args.shard_size)]
    pool: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        pool.put(gpu)
    logs = ROOT / "closure" / "tasks" / args.point / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run(job: tuple[str, int, int]) -> tuple[str, int, int]:
        dataset, begin, end = job
        answer = (ROOT / "closure" / "tasks" / args.point / "shards" / dataset /
                  f"shard_{begin}_{end}" / dataset /
                  f"pareto_{args.point}-fp16.jsonl")
        if complete(answer, end - begin):
            return job
        gpu = pool.get()
        env = os.environ.copy()
        env["BATCH_SIZE"] = str(args.batch_size)
        try:
            with (logs / f"{dataset}_{begin}_{end}_gpu{gpu}.log").open("w") as log:
                subprocess.run(["bash", str(ROOT / "scripts" / "run_pareto_pmpd_shard.sh"),
                                args.point, dataset, str(begin), str(end), gpu],
                               cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        finally:
            pool.put(gpu)
        return job

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            dataset, begin, end = future.result()
            print(f"completed {dataset} [{begin}, {end})", flush=True)


if __name__ == "__main__":
    main()
