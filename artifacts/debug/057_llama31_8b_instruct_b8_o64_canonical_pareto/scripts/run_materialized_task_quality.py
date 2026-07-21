#!/usr/bin/env python3
"""Evaluate phase policies with one temporary checkpoint and parallel GPU shards."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = Path(os.environ["COSPAQ_EXPERIMENT_DIR"])
EXPORTER = Path(os.environ["COSPAQ_EXPORTER"])
RUNNER = ROOT / "artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/scripts/run_task_quality_shard.sh"
PYTHON = "/home/agent/wja/miniconda3/envs/vllm/bin/python"
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
DEFAULT = "p00,p01,p02,p03,p04," + ",".join(f"point_{i:03d}" for i in range(12))


def complete(path: Path, count: int) -> bool:
    return path.exists() and sum(1 for _ in path.open(encoding="utf-8")) == count


def policy_path(policy_id: str) -> Path:
    if policy_id.startswith("point_"):
        return EXP / "pareto/policies" / f"{policy_id}.json"
    return EXP / "policies/prefill_decode" / f"{policy_id}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default=DEFAULT)
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    parser.add_argument("--export-gpu", default="2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--shard-size", type=int, default=360)
    parser.add_argument("--iwslt-shard-size", type=int, default=100)
    parser.add_argument("--force-datasets", default="",
                        help="Comma-separated datasets whose shards must be regenerated.")
    args = parser.parse_args()
    policies = [item for item in args.policies.split(",") if item]
    gpus = [item for item in args.gpus.split(",") if item]
    forced_datasets = {item for item in args.force_datasets.split(",") if item}
    task_root = EXP / "task_quality"

    for policy_id in policies:
        checkpoint = task_root / "checkpoints" / policy_id
        policy = policy_path(policy_id)
        if not policy.exists():
            raise FileNotFoundError(policy)
        if not (checkpoint / "model.safetensors").exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = args.export_gpu
            log = task_root / "logs/export" / f"{policy_id}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as handle:
                subprocess.run([PYTHON, str(EXPORTER), "--policy-json", str(policy),
                                "--output-dir", str(checkpoint), "--force"],
                               env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

        label = f"phase_{policy_id}_prefill_decode"
        pending = []
        for dataset, size in DATASETS.items():
            width = args.iwslt_shard_size if dataset == "IWSLT" else args.shard_size
            for begin in range(0, size, width):
                end = min(begin + width, size)
                shard = task_root / "shards" / policy_id / dataset / f"shard_{begin:04d}_{end:04d}"
                output = shard / dataset / f"{label}-fp16.jsonl"
                if dataset in forced_datasets or not complete(output, end - begin):
                    pending.append((dataset, begin, end, shard))

        available: queue.Queue[str] = queue.Queue()
        for gpu in gpus: available.put(gpu)

        def run(item: tuple[str, int, int, Path]) -> str:
            dataset, begin, end, shard = item
            gpu = available.get()
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": gpu, "CHECKPOINT": str(checkpoint), "DATASET": dataset,
                        "QUESTION_BEGIN": str(begin), "QUESTION_END": str(end), "OUT_DIR": str(shard),
                        "LABEL": label, "BATCH_SIZE": str(args.batch_size),
                        "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
                        "MAX_MODEL_LEN": "4096"})
            log = task_root / "logs/shards" / f"{policy_id}_{dataset}_{begin:04d}_{end:04d}_gpu{gpu}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log.open("w") as handle:
                    subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, check=True)
            finally:
                available.put(gpu)
            return f"{policy_id}/{dataset}[{begin},{end})"

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run, item) for item in pending]):
                print(future.result(), flush=True)
        shutil.rmtree(checkpoint, ignore_errors=True)
        print(f"completed {policy_id}", flush=True)


if __name__ == "__main__":
    main()
