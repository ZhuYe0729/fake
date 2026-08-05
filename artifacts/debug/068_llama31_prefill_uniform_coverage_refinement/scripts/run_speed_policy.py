#!/usr/bin/env python3
"""Measure one policy with the final 061 warmed E2E speed protocol."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess

from scenario import (
    BATCH,
    BENCH,
    CANONICAL,
    COSPAQ_PYTHON,
    COSPAQ_SPARSE_BF16_EXTENSION_DIR,
    COSPAQ_SPARSE_NVFP4_EXTENSION_DIR,
    EXP,
    INPUT_TOKENS,
    MAX_BATCHED_TOKENS,
    MODEL,
    REPEATS,
    REUSE_DENSE_EXPORTER,
    VLLM_NVFP4_EXTENSION_DIR,
    VLLM_PYTHON,
    VLLM_SPARSE_BF16_EXTENSION_DIR,
    VLLM_SPARSE_NVFP4_EXTENSION_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    policy = EXP / "policies" / f"{args.policy}.json"
    if not policy.exists():
        raise FileNotFoundError(policy)
    output = EXP / "speed" / f"{args.policy}.json"
    if output.exists() and not args.force:
        print(output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = EXP / "temporary/speed" / args.policy
    methods = {
        item["prefill_method"]
        for item in json.loads(policy.read_text())["method_map"].values()
    }

    export_env = os.environ.copy()
    export_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    export_env["CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR"] = str(
        COSPAQ_SPARSE_NVFP4_EXTENSION_DIR
    )
    export_env["CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR"] = str(
        COSPAQ_SPARSE_BF16_EXTENSION_DIR
    )
    benchmark_env = export_env.copy()
    benchmark_env["CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR"] = str(
        VLLM_NVFP4_EXTENSION_DIR
    )
    benchmark_env["CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR"] = str(
        VLLM_SPARSE_NVFP4_EXTENSION_DIR
    )
    benchmark_env["CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR"] = str(
        VLLM_SPARSE_BF16_EXTENSION_DIR
    )

    lock = EXP / "temporary/export.lock"
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            shutil.rmtree(checkpoint, ignore_errors=True)
            command = [
                str(COSPAQ_PYTHON),
                str(REUSE_DENSE_EXPORTER),
                "--policy-json",
                str(policy),
                "--model-path",
                str(MODEL),
                "--output-dir",
                str(checkpoint),
            ]
            if "sparse_bf16" in methods:
                command += [
                    "--canonical-sparse-bf16-state",
                    str(CANONICAL / "sparse_bf16/model.pt"),
                ]
            if "sparse_nvfp4" in methods:
                command += [
                    "--canonical-sparse-nvfp4-state",
                    str(CANONICAL / "sparse_nvfp4/model.pt"),
                ]
            command.append("--force")
            subprocess.run(command, check=True, env=export_env)
        subprocess.run(
            [
                str(VLLM_PYTHON),
                str(BENCH),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--batch",
                str(BATCH),
                "--input-seq",
                str(INPUT_TOKENS),
                "--max-num-batched-tokens",
                str(MAX_BATCHED_TOKENS),
                "--warmup",
                "1",
                "--repeats",
                str(REPEATS),
            ],
            check=True,
            env=benchmark_env,
        )
    finally:
        if not args.keep_checkpoint:
            shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
