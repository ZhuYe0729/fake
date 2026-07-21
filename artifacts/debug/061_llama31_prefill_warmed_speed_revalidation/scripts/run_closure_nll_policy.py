#!/usr/bin/env python3
"""Export one solved policy and measure its real phase-vLLM prefill NLL."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys

from scenario import (EXP, SOURCE, MODEL, REUSE_DENSE_EXPORTER, CANONICAL, ROOT,
                      VLLM_NVFP4_EXTENSION_DIR, COSPAQ_SPARSE_NVFP4_EXTENSION_DIR,
                      VLLM_SPARSE_NVFP4_EXTENSION_DIR, COSPAQ_SPARSE_BF16_EXTENSION_DIR,
                      VLLM_SPARSE_BF16_EXTENSION_DIR)

EVALUATOR = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_runtime_prefill_nll.py"
VLLM_PYTHON = "/home/agent/wja/miniconda3/envs/vllm/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--blocks", default=100, type=int)
    args = parser.parse_args()
    policy = EXP / "pareto/policies" / f"{args.policy}.json"
    output = EXP / "pareto/closure/nll" / f"{args.policy}.json"
    if output.exists():
        return
    methods = {item["prefill_method"] for item in json.loads(policy.read_text())["method_map"].values()}
    checkpoint = EXP / "temporary/closure_nll" / args.policy
    export_lock = EXP / "temporary/export.lock"
    export_env = os.environ.copy(); export_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    export_env["CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR"] = str(COSPAQ_SPARSE_NVFP4_EXTENSION_DIR)
    export_env["CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR"] = str(COSPAQ_SPARSE_BF16_EXTENSION_DIR)
    eval_env = export_env.copy()
    eval_env["CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR"] = str(VLLM_NVFP4_EXTENSION_DIR)
    eval_env["CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR"] = str(VLLM_SPARSE_NVFP4_EXTENSION_DIR)
    eval_env["CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR"] = str(VLLM_SPARSE_BF16_EXTENSION_DIR)
    eval_env["COSPAQ_VLLM_NLL_GPU_MEMORY_UTILIZATION"] = "0.70"
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
        subprocess.run([VLLM_PYTHON, str(EVALUATOR), "--checkpoint", str(checkpoint), "--tokenizer", str(MODEL),
                        "--samples", str(SOURCE / "samples/wikitext_2048.pt"), "--output", str(output), "--label", args.policy,
                        "--policy-json", str(policy), "--phase-hetero", "--blocks", str(args.blocks)], check=True, env=eval_env)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
