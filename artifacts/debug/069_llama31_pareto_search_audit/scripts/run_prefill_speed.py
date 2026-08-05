#!/usr/bin/env python3
"""Measure one audit policy with the final warmed prefill protocol."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess

from config import (
    CANONICAL,
    COSPAQ_PYTHON,
    COSPAQ_SPARSE_BF16_EXTENSION,
    COSPAQ_SPARSE_NVFP4_EXTENSION,
    DEBUG,
    MARLIN_EXTENSION,
    MODEL,
    PREFILL,
    PREFILL_BENCH,
    PREFILL_EXPORTER,
    VLLM_NVFP4_EXTENSION,
    VLLM_PYTHON,
    VLLM_SPARSE_BF16_EXTENSION,
    VLLM_SPARSE_NVFP4_EXTENSION,
    RUNTIME_TMP,
    candidate_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--policy-json")
    parser.add_argument("--output-name")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    policy = (
        candidate_path(PREFILL, args.policy)
        if args.policy_json is None
        else __import__("pathlib").Path(args.policy_json)
    )
    output_name = args.output_name or args.policy
    output = PREFILL / f"speed/{output_name}.json"
    if output.exists() and not args.force:
        print(output)
        return
    RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
    stage_lock = DEBUG / "temporary/speed_stage.lock"
    stage_lock.parent.mkdir(parents=True, exist_ok=True)
    stage_handle = stage_lock.open("w")
    fcntl.flock(stage_handle, fcntl.LOCK_EX)
    run_lock = PREFILL / f"temporary/locks/speed_{output_name}.lock"
    run_lock.parent.mkdir(parents=True, exist_ok=True)
    run_handle = run_lock.open("w")
    fcntl.flock(run_handle, fcntl.LOCK_EX)
    if output.exists() and not args.force:
        print(output)
        run_handle.close()
        stage_handle.close()
        return
    checkpoint = PREFILL / f"temporary/speed/{output_name}"
    methods = {item["prefill_method"] for item in json.loads(policy.read_text())["method_map"].values()}
    export_env = os.environ.copy()
    export_env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "TMPDIR": str(RUNTIME_TMP),
        "CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR": str(MARLIN_EXTENSION),
        "CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR": str(COSPAQ_SPARSE_NVFP4_EXTENSION),
        "CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR": str(COSPAQ_SPARSE_BF16_EXTENSION),
    })
    bench_env = export_env.copy()
    bench_env.update({
        "CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR": str(VLLM_NVFP4_EXTENSION),
        "CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR": str(VLLM_SPARSE_NVFP4_EXTENSION),
        "CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR": str(VLLM_SPARSE_BF16_EXTENSION),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = PREFILL / "temporary/export.lock"
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            shutil.rmtree(checkpoint, ignore_errors=True)
            command = [
                str(COSPAQ_PYTHON), str(PREFILL_EXPORTER),
                "--policy-json", str(policy), "--model-path", str(MODEL),
                "--output-dir", str(checkpoint),
            ]
            if "sparse_bf16" in methods:
                command += ["--canonical-sparse-bf16-state", str(CANONICAL / "sparse_bf16/model.pt")]
            if "sparse_nvfp4" in methods:
                command += ["--canonical-sparse-nvfp4-state", str(CANONICAL / "sparse_nvfp4/model.pt")]
            command.append("--force")
            subprocess.run(command, check=True, env=export_env)
        subprocess.run([
            str(VLLM_PYTHON), str(PREFILL_BENCH), "--checkpoint", str(checkpoint),
            "--output", str(output), "--batch", "8", "--input-seq", "2048",
            "--max-num-batched-tokens", "16384", "--warmup", "1", "--repeats", "5",
        ], check=True, env=bench_env)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)
        run_handle.close()
        stage_handle.close()


if __name__ == "__main__":
    main()
