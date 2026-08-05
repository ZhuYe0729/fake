#!/usr/bin/env python3
"""Materialize one phase policy and evaluate the frozen IWSLT-333 task."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

from config import (
    DECODE,
    DECODE_EXPORTER,
    MARLIN_EXTENSION,
    MODEL,
    PMPD_EVAL,
    TASK_SHARD_RUNNER,
    RUNTIME_TMP,
    VLLM_PYTHON,
    candidate_path,
)


def complete(path: Path, expected: int) -> bool:
    return path.exists() and sum(1 for _ in path.open(encoding="utf-8")) == expected


def run_shard(checkpoint: Path, policy_id: str, gpu: str, begin: int, end: int, batch: int, memory: float) -> Path:
    shard = DECODE / f"quality/shards/{policy_id}/IWSLT/shard_{begin:04d}_{end:04d}"
    label = f"phase_{policy_id}_prefill_decode"
    output = shard / "IWSLT" / f"{label}-fp16.jsonl"
    if complete(output, end - begin):
        return output
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu, "CHECKPOINT": str(checkpoint), "DATASET": "IWSLT",
        "QUESTION_BEGIN": str(begin), "QUESTION_END": str(end), "OUT_DIR": str(shard),
        "LABEL": label, "BATCH_SIZE": str(batch), "GPU_MEMORY_UTILIZATION": str(memory),
        "MAX_MODEL_LEN": "4096", "KV_CACHE_DTYPE": "auto",
        "TMPDIR": str(RUNTIME_TMP),
        "CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR": str(MARLIN_EXTENSION),
        "PYTHON_BIN": str(VLLM_PYTHON),
    })
    subprocess.run(["bash", str(TASK_SHARD_RUNNER)], check=True, env=env)
    if not complete(output, end - begin):
        raise RuntimeError(f"incomplete shard {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    metrics_file = DECODE / f"quality/results/{args.policy}/IWSLT/metrics.json"
    if metrics_file.exists() and not args.force:
        print(metrics_file)
        return
    RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
    run_lock = DECODE / f"temporary/locks/quality_{args.policy}.lock"
    run_lock.parent.mkdir(parents=True, exist_ok=True)
    run_handle = run_lock.open("w")
    fcntl.flock(run_handle, fcntl.LOCK_EX)
    if metrics_file.exists() and not args.force:
        print(metrics_file)
        run_handle.close()
        return
    policy = candidate_path(DECODE, args.policy)
    checkpoint = DECODE / f"temporary/quality/{args.policy}"
    try:
        if not (checkpoint / "config.json").exists():
            export_lock = DECODE / "temporary/export.lock"
            export_lock.parent.mkdir(parents=True, exist_ok=True)
            with export_lock.open("w") as export_handle:
                fcntl.flock(export_handle, fcntl.LOCK_EX)
                if not (checkpoint / "config.json").exists():
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    env.update({
                        "CUDA_VISIBLE_DEVICES": args.gpu,
                        "TMPDIR": str(RUNTIME_TMP),
                        "CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR": str(MARLIN_EXTENSION),
                    })
                    subprocess.run([
                        str(VLLM_PYTHON), str(DECODE_EXPORTER), "--policy-json", str(policy),
                        "--output-dir", str(checkpoint), "--model-path", str(MODEL), "--force",
                    ], check=True, env=env)
        outputs = []
        for begin, end in ((0, 100), (100, 200), (200, 300), (300, 333)):
            try:
                outputs.append(run_shard(checkpoint, args.policy, args.gpu, begin, end, 4, 0.75))
            except subprocess.CalledProcessError:
                outputs.append(run_shard(checkpoint, args.policy, args.gpu, begin, end, 1, 0.60))
        records = {}
        for path in outputs:
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                key = str(item["question_id"])
                if key in records:
                    raise RuntimeError(f"duplicate IWSLT id {key}")
                records[key] = item
        if len(records) != 333:
            raise RuntimeError(f"IWSLT rows={len(records)}")
        result = DECODE / f"quality/results/{args.policy}/IWSLT/phase_{args.policy}_prefill_decode-fp16.jsonl"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text("".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in sorted(records.values(), key=lambda value: int(value["question_id"]))
        ), encoding="utf-8")
        subprocess.run([
            str(VLLM_PYTHON), str(PMPD_EVAL), "--dataset", "IWSLT",
            "--metrics-only", str(result),
        ], check=True)
        generated_metrics = result.with_name("metrics.json")
        if generated_metrics != metrics_file:
            shutil.copy2(generated_metrics, metrics_file)
        print(metrics_file)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)
        run_handle.close()


if __name__ == "__main__":
    main()
