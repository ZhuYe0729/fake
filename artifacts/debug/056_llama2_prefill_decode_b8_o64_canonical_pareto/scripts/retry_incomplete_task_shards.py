#!/usr/bin/env python3
"""Retry only OOM-truncated task tails with one request per vLLM batch."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
EXP = ROOT / "llama2_7b_chat"
RUNNER = REPO / "artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/scripts/run_task_quality_shard.sh"
RETRIES = (
    ("5", "b8o64009", "cnn_dm_1000", 884, 1000, "0.50"),
)


def main() -> None:
    def run(item: tuple[str, str, str, int, int, str]) -> str:
        gpu, policy, dataset, begin, end, memory_utilization = item
        shard = EXP / "task_quality/shards" / policy / dataset / f"shard_{begin:04d}_{end:04d}_retry_b1"
        target = shard / dataset / f"ours_{policy}_prefill_decode-fp16.jsonl"
        if target.exists() and sum(1 for _ in target.open()) == end - begin:
            return policy
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": gpu,
                    "CHECKPOINT": str(EXP / "speed/runs" / policy / "checkpoint"),
                    "DATASET": dataset, "QUESTION_BEGIN": str(begin), "QUESTION_END": str(end),
                    "OUT_DIR": str(shard), "LABEL": f"ours_{policy}_prefill_decode",
                    "BATCH_SIZE": "1", "GPU_MEMORY_UTILIZATION": memory_utilization})
        with (EXP / "task_quality/logs" / f"retry_{policy}_{dataset}_{begin:04d}_{end:04d}.log").open("w") as handle:
            subprocess.run(["bash", str(RUNNER)], cwd=REPO, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        return policy

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RETRIES)) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, item) for item in RETRIES]):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
