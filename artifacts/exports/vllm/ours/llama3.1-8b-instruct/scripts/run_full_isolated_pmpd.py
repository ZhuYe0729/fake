#!/usr/bin/env python3
"""Run Llama3.1 max-speed PMPD shards on GPUs 5, 6, and 7 only."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
DATASET_SIZES = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
SCENARIOS = ("prefill_only", "prefill_decode")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="5,6,7")
    p.add_argument("--shard-size", type=int, default=360)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--python", default="/home/agent/wja/miniconda3/envs/vllm/bin/python")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus or any(x not in {"5", "6", "7"} for x in gpus):
        raise ValueError("--gpus must be a non-empty subset of 5,6,7")
    tasks = [(scenario, dataset, begin, min(begin + args.shard_size, size)) for scenario in SCENARIOS for dataset, size in DATASET_SIZES.items() for begin in range(0, size, args.shard_size)]
    logs = ROOT / "max_speed/quality_jobs"; logs.mkdir(parents=True, exist_ok=True)
    available: queue.Queue[str] = queue.Queue()
    for gpu in gpus: available.put(gpu)
    def run(task: tuple[str, str, int, int]) -> tuple[str, str, int, int]:
        scenario, dataset, begin, end = task
        shard_root = ROOT / "max_speed" / scenario / "quality_shards" / dataset / f"shard_{begin:04d}_{end:04d}"
        existing = shard_root / dataset / f"ours_max_speed_{scenario}-fp16.jsonl"
        if existing.exists() and sum(1 for _ in existing.open(encoding="utf-8")) == end - begin:
            return task
        gpu = available.get()
        env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": gpu, "SCENARIO": scenario, "DATASET": dataset, "QUESTION_BEGIN": str(begin), "QUESTION_END": str(end), "BATCH_SIZE": str(args.batch_size), "OUT_DIR": str(shard_root), "LABEL": f"ours_max_speed_{scenario}", "SKIP_FINAL_METRICS": "1"})
        try:
            with (logs / f"{scenario}_{dataset}_{begin:04d}_{end:04d}_gpu{gpu}.log").open("w") as log:
                subprocess.run(["bash", str(SCRIPT_DIR / "run_isolated_pmpd.sh")], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            available.put(gpu)
        return task
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, task) for task in tasks]):
            scenario, dataset, begin, end = future.result()
            print(f"completed {scenario}/{dataset} [{begin},{end})", flush=True)
    merger = ROOT / "../llama2-7b-chat/scripts/merge_pmpd_shards.py"
    for scenario in SCENARIOS:
        for dataset in DATASET_SIZES:
            subprocess.run([args.python, str(merger), "--shard-root", str(ROOT / "max_speed" / scenario / "quality_shards" / dataset), "--output-dir", str(ROOT / "max_speed" / scenario / "results/quality"), "--dataset", dataset, "--label", f"ours_max_speed_{scenario}", "--python", args.python], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
