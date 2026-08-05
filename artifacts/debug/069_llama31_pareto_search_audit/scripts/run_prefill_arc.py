#!/usr/bin/env python3
"""Evaluate one audit policy on full ARC-Challenge with the frozen runtime."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess

from config import (
    CANONICAL,
    COSPAQ_PYTHON,
    COSPAQ_SPARSE_BF16_EXTENSION,
    COSPAQ_SPARSE_NVFP4_EXTENSION,
    MARLIN_EXTENSION,
    MODEL,
    PREFILL,
    PREFILL_EVALUATOR,
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    policy = candidate_path(PREFILL, args.policy)
    result = PREFILL / f"quality/{args.policy}/arc_challenge/full/result.json"
    if result.exists() and not args.force:
        print(result)
        return
    RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
    run_lock = PREFILL / f"temporary/locks/quality_{args.policy}.lock"
    run_lock.parent.mkdir(parents=True, exist_ok=True)
    run_handle = run_lock.open("w")
    fcntl.flock(run_handle, fcntl.LOCK_EX)
    if result.exists() and not args.force:
        print(result)
        run_handle.close()
        return
    checkpoint = PREFILL / f"temporary/quality/{args.policy}"
    methods = {item["prefill_method"] for item in json.loads(policy.read_text())["method_map"].values()}
    export_env = os.environ.copy()
    export_env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "TMPDIR": str(RUNTIME_TMP),
        "CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR": str(MARLIN_EXTENSION),
        "CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR": str(COSPAQ_SPARSE_NVFP4_EXTENSION),
        "CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR": str(COSPAQ_SPARSE_BF16_EXTENSION),
    })
    task_env = export_env.copy()
    task_env.update({
        "TOKENIZERS_PARALLELISM": "false", "COSPAQ_TASK_SAFE_MODE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR": str(VLLM_NVFP4_EXTENSION),
        "CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR": str(VLLM_SPARSE_NVFP4_EXTENSION),
        "CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR": str(VLLM_SPARSE_BF16_EXTENSION),
    })
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
        manifest = PREFILL / f"quality/manifests/{args.policy}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "model_path": str(MODEL),
            "policies": [{
                "label": args.policy, "kind": "ours", "checkpoint": str(checkpoint),
                "policy_json": str(policy),
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            }],
        }, indent=2) + "\n")
        subprocess.run([
            str(VLLM_PYTHON), str(PREFILL_EVALUATOR), "--manifest", str(manifest),
            "--policy", args.policy, "--output-root", str(PREFILL / "quality"),
        ], check=True, env=task_env)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)
        run_handle.close()


if __name__ == "__main__":
    main()
