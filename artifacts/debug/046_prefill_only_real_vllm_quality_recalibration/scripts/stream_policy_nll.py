#!/usr/bin/env python3
"""Score one policy and remove temporary phase checkpoints afterwards."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import CUTLASS, MODELS, VLLM_ROOT, model_root, normalized_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--blocks", type=int, default=100)
    return parser.parse_args()


def uniform_checkpoint(model: str, policy: str) -> Path:
    root = MODELS[model]["uniform_root"]
    if policy == "p00":
        return MODELS[model]["path"]
    names = {"p01": "uniform_dense_nvfp4", "p02": "uniform_sparse_bf16", "p03": "uniform_sparse_nvfp4", "p04": "uniform_marlin_nvfp4"}
    return root / names[policy]


def main() -> None:
    args = parse_args()
    root = model_root(args.model)
    policy_path = root / "policies/prefill_only" / f"{args.policy}.json"
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)
    output = root / "results" / f"{args.policy}.json"
    if output.exists():
        raise FileExistsError(output)
    samples = root / "samples/wikitext_2048_targets.pt"
    evaluator = Path(__file__).with_name("evaluate_runtime_prefill_nll.py")
    if args.policy in {"p00", "p01", "p02", "p03", "p04"}:
        checkpoint = uniform_checkpoint(args.model, args.policy)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        command = [sys.executable, str(evaluator), "--checkpoint", str(checkpoint), "--tokenizer", str(MODELS[args.model]["path"]), "--samples", str(samples), "--output", str(output), "--label", args.policy, "--blocks", str(args.blocks)]
        subprocess.run(command, check=True)
        return

    # Exports are transient and can exceed the repository's remaining space.
    # Keep them outside the experiment bundle when the dispatcher provides a
    # scratch root; results and manifests always remain under debug 046.
    temporary_root = Path(os.environ.get("COSPAQ_PHASE_TMP_ROOT", str(root / "temporary_checkpoints")))
    checkpoint = temporary_root / args.model / args.policy
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    policy = normalized_policy(policy_path)
    methods = [policy["default_prefill_method"], policy["default_decode_method"]]
    methods.extend(pair[phase] for pair in policy["method_map"].values() for phase in ("prefill_method", "decode_method"))
    exporter = VLLM_ROOT / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
    command = [sys.executable, str(exporter), "--model-path", str(MODELS[args.model]["path"]), "--policy-json", str(policy_path), "--output-dir", str(checkpoint), "--cutlass-wrapper-path", str(CUTLASS)]
    if any(method.startswith("sparse_") for method in methods):
        command.append("--prune")
    try:
        subprocess.run(command, check=True)
        if normalized_policy(checkpoint / "phase_hetero_policy.json") != policy:
            raise RuntimeError("exported policy differs from source JSON")
        subprocess.run([sys.executable, str(evaluator), "--checkpoint", str(checkpoint), "--tokenizer", str(MODELS[args.model]["path"]), "--samples", str(samples), "--output", str(output), "--label", args.policy, "--policy-json", str(policy_path), "--phase-hetero", "--blocks", str(args.blocks)], check=True)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
