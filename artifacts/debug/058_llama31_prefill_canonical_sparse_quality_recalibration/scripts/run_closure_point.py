#!/usr/bin/env python3
"""Close one solved mixed policy with real phase-vLLM NLL and five E2E samples."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from scenario import EXP, MODEL, ROOT, BATCH, INPUT_TOKENS

EXPORT = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/scripts/export_phase_checkpoint.py"
EVALUATOR = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_runtime_prefill_nll.py"
BENCH = ROOT / "artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py"
VLLM_PYTHON = Path("/home/agent/wja/miniconda3/envs/vllm/bin/python")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--policy", required=True); parser.add_argument("--gpu", type=int, required=True); args = parser.parse_args()
    policy = EXP / "pareto/policies" / f"{args.policy}.json"; root = EXP / "pareto/closure"; nll = root / "nll" / f"{args.policy}.json"; runs = root / "speed" / args.policy / "runs"
    if not policy.exists(): raise FileNotFoundError(policy)
    if nll.exists() and all((runs / f"measured_{i}.json").exists() for i in range(5)): return
    checkpoint = EXP / "temporary/closure" / args.policy
    if checkpoint.exists(): shutil.rmtree(checkpoint)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu); env.setdefault("COSPAQ_VLLM_NLL_GPU_MEMORY_UTILIZATION", "0.70")
    try:
        subprocess.run([sys.executable, str(EXPORT), "--policy-json", str(policy), "--model-path", str(MODEL), "--output-dir", str(checkpoint), "--force"], check=True, env=env)
        nll.parent.mkdir(parents=True, exist_ok=True)
        if not nll.exists():
            subprocess.run([str(VLLM_PYTHON), str(EVALUATOR), "--checkpoint", str(checkpoint), "--tokenizer", str(MODEL), "--samples", str(EXP / "samples/wikitext_2048.pt"), "--output", str(nll), "--label", args.policy, "--policy-json", str(policy), "--phase-hetero", "--blocks", "100"], check=True, env=env)
        runs.mkdir(parents=True, exist_ok=True)
        for tag in ("warmup", *(f"measured_{i}" for i in range(5))):
            result = runs / f"{tag}.json"
            if not result.exists():
                subprocess.run([str(VLLM_PYTHON), str(BENCH), "--checkpoint", str(checkpoint), "--batch", str(BATCH), "--input-seq", str(INPUT_TOKENS), "--output-seq", "1", "--gpu-memory-utilization", "0.90", "--output-json", str(result)], check=True, env=env)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)
if __name__ == "__main__": main()
