#!/usr/bin/env python3
"""Materialize one canonical policy and record one warmup plus five E2E runs."""
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
BENCH = ROOT / "artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py"
VLLM_PYTHON = Path("/home/agent/wja/miniconda3/envs/vllm/bin/python")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--policy", required=True); parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--keep-checkpoint", action="store_true"); args = parser.parse_args()
    policy = EXP / "policies/prefill_only" / f"{args.policy}.json"
    if not policy.exists(): raise FileNotFoundError(policy)
    base = EXP / "speed/calibration"; runs = base / "runs" / args.policy; runs.mkdir(parents=True, exist_ok=True)
    expected = [runs / f"measured_{i}.json" for i in range(5)]
    if all(path.exists() for path in expected): return
    # /tmp is a small overlay filesystem here; use the experiment filesystem
    # so concurrent, disposable 8B exports cannot exhaust the overlay.
    checkpoint = EXP / "temporary/speed" / args.policy
    if checkpoint.exists(): shutil.rmtree(checkpoint)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        subprocess.run([sys.executable, str(EXPORT), "--policy-json", str(policy), "--model-path", str(MODEL),
                        "--output-dir", str(checkpoint), "--force"], check=True, env=env)
        for tag in ("warmup", *(f"measured_{i}" for i in range(5))):
            result = runs / f"{tag}.json"
            if result.exists(): continue
            subprocess.run([str(VLLM_PYTHON), str(BENCH), "--checkpoint", str(checkpoint),
                            "--batch", str(BATCH), "--input-seq", str(INPUT_TOKENS), "--output-seq", "1",
                            "--gpu-memory-utilization", "0.90", "--output-json", str(result)], check=True, env=env)
    finally:
        if not args.keep_checkpoint: shutil.rmtree(checkpoint, ignore_errors=True)
if __name__ == "__main__": main()
