#!/usr/bin/env python3
"""Run frozen real-vLLM prefill tasks for one newly solved policy."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys

from scenario import (EXP, MODEL, ROOT, REUSE_DENSE_EXPORTER, CANONICAL,
                      VLLM_NVFP4_EXTENSION_DIR, COSPAQ_SPARSE_NVFP4_EXTENSION_DIR,
                      VLLM_SPARSE_NVFP4_EXTENSION_DIR, COSPAQ_SPARSE_BF16_EXTENSION_DIR,
                      VLLM_SPARSE_BF16_EXTENSION_DIR)

EVALUATOR = ROOT / "artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality/scripts/evaluate_policy.py"
VLLM_PYTHON = "/home/agent/wja/miniconda3/envs/vllm/bin/python"
TASKS = ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    policy = EXP / "pareto/policies" / f"{args.policy}.json"
    output = EXP / "task_quality/results" / args.policy
    if all((output / task / "full/result.json").exists() for task in TASKS):
        return
    methods = {item["prefill_method"] for item in json.loads(policy.read_text())["method_map"].values()}
    checkpoint = EXP / "temporary/task" / args.policy
    export_lock = EXP / "temporary/export.lock"
    export_env = os.environ.copy(); export_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    export_env["CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR"] = str(COSPAQ_SPARSE_NVFP4_EXTENSION_DIR)
    export_env["CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR"] = str(COSPAQ_SPARSE_BF16_EXTENSION_DIR)
    task_env = export_env.copy()
    task_env.update({"TOKENIZERS_PARALLELISM": "false", "COSPAQ_TASK_SAFE_MODE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
                     "CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR": str(VLLM_NVFP4_EXTENSION_DIR),
                     "CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR": str(VLLM_SPARSE_NVFP4_EXTENSION_DIR),
                     "CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR": str(VLLM_SPARSE_BF16_EXTENSION_DIR)})
    try:
        export_lock.parent.mkdir(parents=True, exist_ok=True)
        with export_lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
            command = [sys.executable, str(REUSE_DENSE_EXPORTER), "--policy-json", str(policy), "--model-path", str(MODEL),
                       "--output-dir", str(checkpoint)]
            if "sparse_bf16" in methods:
                command += ["--canonical-sparse-bf16-state", str(CANONICAL / "sparse_bf16/model.pt")]
            if "sparse_nvfp4" in methods:
                command += ["--canonical-sparse-nvfp4-state", str(CANONICAL / "sparse_nvfp4/model.pt")]
            command.append("--force")
            subprocess.run(command, check=True, env=export_env)
            fcntl.flock(handle, fcntl.LOCK_UN)
        manifest = {"model_path": str(MODEL), "policies": [{"label": args.policy, "kind": "ours", "checkpoint": str(checkpoint),
                     "policy_json": str(policy), "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}]}
        manifest_path = EXP / "task_quality/manifests" / f"{args.policy}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run([VLLM_PYTHON, str(EVALUATOR), "--manifest", str(manifest_path), "--policy", args.policy,
                        "--output-root", str(EXP / "task_quality/results")], check=True, env=task_env)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
