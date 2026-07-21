#!/usr/bin/env python3
"""Export one canonical prefill policy and measure full-prompt phase-vLLM NLL."""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from scenario import EXP, MODEL, ROOT

EXPORT = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/scripts/export_phase_checkpoint.py"
EVALUATOR = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_runtime_prefill_nll.py"
VLLM_PYTHON = Path("/home/agent/wja/miniconda3/envs/vllm/bin/python")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--policy", required=True); parser.add_argument("--gpu", type=int, required=True); args = parser.parse_args()
    policy = EXP / "policies/prefill_only" / f"{args.policy}.json"; output = EXP / "nll/raw" / f"{args.policy}.json"
    if output.exists(): return
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = EXP / "logs/nll"; logs.mkdir(parents=True, exist_ok=True)
    checkpoint = EXP / "temporary/nll" / args.policy
    if checkpoint.exists(): shutil.rmtree(checkpoint)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("COSPAQ_VLLM_NLL_GPU_MEMORY_UTILIZATION", "0.70")
    try:
        with (logs / f"{args.policy}.export.log").open("w") as log:
            subprocess.run([sys.executable, str(EXPORT), "--policy-json", str(policy), "--model-path", str(MODEL), "--output-dir", str(checkpoint), "--force"], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        with (logs / f"{args.policy}.eval.log").open("w") as log:
            subprocess.run([str(VLLM_PYTHON), str(EVALUATOR), "--checkpoint", str(checkpoint), "--tokenizer", str(MODEL),
                            "--samples", str(EXP / "samples/wikitext_2048.pt"), "--output", str(output), "--label", args.policy,
                            "--policy-json", str(policy), "--phase-hetero", "--blocks", "100"], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)
if __name__ == "__main__": main()
