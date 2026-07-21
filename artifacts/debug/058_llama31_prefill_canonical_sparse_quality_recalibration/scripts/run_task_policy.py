#!/usr/bin/env python3
"""Run five real-vLLM prefill tasks for one canonical uniform or mixed policy."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from scenario import EXP, MODEL, ROOT

EXPORT = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/scripts/export_phase_checkpoint.py"
EVALUATOR = ROOT / "artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality/scripts/evaluate_policy.py"
VLLM_PYTHON = Path("/home/agent/wja/miniconda3/envs/vllm/bin/python")
TASKS = ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--policy-json", type=Path, required=True); parser.add_argument("--label", required=True); parser.add_argument("--gpu", type=int, required=True); args = parser.parse_args()
    output = EXP / "task_quality/results" / args.label
    if all((output / task / "full/result.json").exists() for task in TASKS): return
    locks = EXP / "task_quality/locks"; locks.mkdir(parents=True, exist_ok=True)
    lock = locks / f"{args.label}.lock"
    try:
        with lock.open("x") as handle:
            handle.write(f"gpu={args.gpu}\n")
    except FileExistsError:
        # Another scheduler is already materializing/evaluating this policy.
        return
    checkpoint = EXP / "temporary/task" / args.label
    if checkpoint.exists(): shutil.rmtree(checkpoint)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu); env["TOKENIZERS_PARALLELISM"] = "false"
    # Task quality is not a timing measurement. Use the conservative vLLM
    # allocation on 32-GB cards and never let datasets revalidate cached MMLU
    # metadata over the unreliable external connection.
    env["COSPAQ_TASK_SAFE_MODE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    try:
        subprocess.run([sys.executable, str(EXPORT), "--policy-json", str(args.policy_json), "--model-path", str(MODEL), "--output-dir", str(checkpoint), "--force"], check=True, env=env)
        manifest = {"model_path": str(MODEL), "policies": [{"label": args.label, "kind": "ours", "checkpoint": str(checkpoint), "policy_json": str(args.policy_json), "policy_sha256": hashlib.sha256(args.policy_json.read_bytes()).hexdigest()}]}
        manifest_path = EXP / "task_quality/manifests" / f"{args.label}.json"; manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run([str(VLLM_PYTHON), str(EVALUATOR), "--manifest", str(manifest_path), "--policy", args.label, "--output-root", str(EXP / "task_quality/results")], check=True, env=env)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)
        lock.unlink(missing_ok=True)
if __name__ == "__main__": main()
