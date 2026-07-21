#!/usr/bin/env python3
"""Export one existing policy and measure it once with the fixed warmed protocol."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from scenario import (EXP, SOURCE, MODEL, REUSE_DENSE_EXPORTER, CANONICAL, BENCH, VLLM_PYTHON,
                      VLLM_NVFP4_EXTENSION_DIR, COSPAQ_SPARSE_NVFP4_EXTENSION_DIR,
                      VLLM_SPARSE_NVFP4_EXTENSION_DIR, COSPAQ_SPARSE_BF16_EXTENSION_DIR,
                      VLLM_SPARSE_BF16_EXTENSION_DIR, BATCH, INPUT_TOKENS, REPEATS,
                      MAX_BATCHED_TOKENS)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--policy", required=True); parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--keep-checkpoint", action="store_true"); args = parser.parse_args()
    output = EXP / "speed/calibration/runs" / f"{args.policy}.json"
    if output.exists(): return
    policy = SOURCE / "policies/prefill_only" / f"{args.policy}.json"
    if not policy.exists(): raise FileNotFoundError(policy)
    methods = {item["prefill_method"] for item in json.loads(policy.read_text())["method_map"].values()}
    checkpoint = EXP / "temporary" / args.policy
    export_lock = EXP / "temporary/export.lock"
    export_env = os.environ.copy(); export_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    export_env["CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR"] = str(COSPAQ_SPARSE_NVFP4_EXTENSION_DIR)
    export_env["CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR"] = str(COSPAQ_SPARSE_BF16_EXTENSION_DIR)
    benchmark_env = export_env.copy()
    benchmark_env["CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR"] = str(VLLM_NVFP4_EXTENSION_DIR)
    benchmark_env["CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR"] = str(VLLM_SPARSE_NVFP4_EXTENSION_DIR)
    benchmark_env["CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR"] = str(VLLM_SPARSE_BF16_EXTENSION_DIR)
    try:
        export_lock.parent.mkdir(parents=True, exist_ok=True)
        with export_lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if checkpoint.exists(): shutil.rmtree(checkpoint)
            command = [sys.executable, str(REUSE_DENSE_EXPORTER), "--model-path", str(MODEL), "--policy-json", str(policy),
                       "--output-dir", str(checkpoint)]
            if "sparse_bf16" in methods:
                command += ["--canonical-sparse-bf16-state", str(CANONICAL / "sparse_bf16/model.pt")]
            if "sparse_nvfp4" in methods:
                command += ["--canonical-sparse-nvfp4-state", str(CANONICAL / "sparse_nvfp4/model.pt")]
            command.append("--force")
            subprocess.run(command, check=True, env=export_env)
            fcntl.flock(handle, fcntl.LOCK_UN)
        subprocess.run([str(VLLM_PYTHON), str(BENCH), "--checkpoint", str(checkpoint), "--output", str(output),
                        "--batch", str(BATCH), "--input-seq", str(INPUT_TOKENS),
                        "--max-num-batched-tokens", str(MAX_BATCHED_TOKENS), "--warmup", "1", "--repeats", str(REPEATS)],
                       check=True, env=benchmark_env)
    finally:
        if not args.keep_checkpoint: shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__": main()
