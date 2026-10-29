#!/usr/bin/env python3
"""Export one phase policy temporarily, measure real vLLM NLL, then remove it."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--policy-json", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--canonical-sparse-bf16-state", type=Path)
    parser.add_argument("--canonical-sparse-nvfp4-state", type=Path)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--cutlass-wrapper-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    policy = json.loads(args.policy_json.read_text())
    checkpoint = args.output.parent / "temporary_checkpoints" / args.label
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    exporter = args.vllm_root / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
    methods = [policy["default_prefill_method"], policy["default_decode_method"]]
    methods.extend(method for pair in policy["method_map"].values() for method in pair.values())
    export = [sys.executable, str(exporter), "--model-path", str(args.model_path),
              "--policy-json", str(args.policy_json), "--output-dir", str(checkpoint),
              "--cutlass-wrapper-path", str(args.cutlass_wrapper_path)]
    if "sparse_bf16" in methods and args.canonical_sparse_bf16_state is None:
        raise ValueError("sparse_bf16 requires --canonical-sparse-bf16-state")
    if "sparse_nvfp4" in methods and args.canonical_sparse_nvfp4_state is None:
        raise ValueError("sparse_nvfp4 requires --canonical-sparse-nvfp4-state")
    if args.canonical_sparse_bf16_state:
        export.extend(["--canonical-sparse-bf16-state", str(args.canonical_sparse_bf16_state)])
    if args.canonical_sparse_nvfp4_state:
        export.extend(["--canonical-sparse-nvfp4-state", str(args.canonical_sparse_nvfp4_state)])
    try:
        subprocess.run(export, check=True)
        if json.loads((checkpoint / "phase_hetero_policy.json").read_text()) != policy:
            raise RuntimeError("exported policy differs from source JSON")
        provenance = checkpoint / "phase_hetero_export_provenance.json"
        if provenance.exists():
            shutil.copy2(provenance, args.output.with_suffix(".export_provenance.json"))
        command = [sys.executable, str(HERE / "evaluate_prefill_decode_nll.py"),
                   "--model", str(checkpoint), "--tokenizer", str(args.tokenizer),
                   "--samples", str(args.samples), "--output", str(args.output),
                   "--label", args.label, "--blocks", str(args.blocks), "--phase-hetero"]
        command.extend(["--input-tokens", str(args.input_tokens),
                        "--output-tokens", str(args.output_tokens),
                        "--batch-size", str(args.batch_size),
                        "--gpu-memory-utilization", str(args.gpu_memory_utilization)])
        command.extend(["--vllm-root", str(args.vllm_root),
                        "--cutlass-wrapper-path", str(args.cutlass_wrapper_path)])
        subprocess.run(command, check=True)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
