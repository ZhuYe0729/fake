#!/usr/bin/env python3
"""Temporarily export one solved policy, then measure real NLL and speed."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import CUTLASS, MODELS, ROOT, VLLM_ROOT, model_root, normalized_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temporary-root", type=Path, default=Path("/tmp/cospaq_phase_pareto_046"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = model_root("llama2")
    label = f"point_{args.point:03d}"
    policy_path = root / "pareto/policies" / f"{label}.json"
    output = root / "pareto/validation"
    nll_output = output / "nll" / f"{label}.json"
    runs = output / "speed" / label / "runs"
    if nll_output.exists() and all((runs / f"measured_{index}.json").exists() for index in range(args.runs)):
        print(f"already complete: {label}")
        return
    policy = normalized_policy(policy_path)
    checkpoint = args.temporary_root / label
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    methods = [policy["default_prefill_method"], policy["default_decode_method"]]
    methods.extend(pair[phase] for pair in policy["method_map"].values() for phase in ("prefill_method", "decode_method"))
    exporter = VLLM_ROOT / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
    evaluator = Path(__file__).with_name("evaluate_runtime_prefill_nll.py")
    benchmark = ROOT / "artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py"
    export = [sys.executable, str(exporter), "--model-path", str(MODELS["llama2"]["path"]), "--policy-json", str(policy_path), "--output-dir", str(checkpoint), "--cutlass-wrapper-path", str(CUTLASS)]
    if any(method.startswith("sparse_") for method in methods):
        export.append("--prune")
    try:
        subprocess.run(export, check=True)
        if normalized_policy(checkpoint / "phase_hetero_policy.json") != policy:
            raise RuntimeError("exported policy differs from source JSON")
        if not nll_output.exists():
            subprocess.run([sys.executable, str(evaluator), "--checkpoint", str(checkpoint), "--tokenizer", str(MODELS["llama2"]["path"]), "--samples", str(root / "samples/wikitext_2048_targets.pt"), "--output", str(nll_output), "--label", label, "--policy-json", str(policy_path), "--phase-hetero", "--blocks", str(args.blocks)], check=True)
        runs.mkdir(parents=True, exist_ok=True)
        for name in ["warmup", *[f"measured_{index}" for index in range(args.runs)]]:
            target = runs / f"{name}.json"
            if target.exists():
                continue
            subprocess.run([sys.executable, str(benchmark), "--checkpoint", str(checkpoint), "--output-json", str(target)], check=True)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
