#!/usr/bin/env python3
"""Export one policy through the external phase-vLLM exporter."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from daso.policy import load_policy, policy_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--cutlass-wrapper-path", type=Path, required=True)
    parser.add_argument("--canonical-sparse-bf16-state", type=Path, required=True)
    parser.add_argument("--canonical-sparse-nvfp4-state", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    policy = load_policy(args.policy_json)
    exporter = args.vllm_root / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
    if not exporter.is_file():
        raise FileNotFoundError(exporter)
    command = [sys.executable, str(exporter), "--model-path", str(args.model_path),
               "--policy-json", str(args.policy_json), "--output-dir", str(args.output_dir),
               "--cutlass-wrapper-path", str(args.cutlass_wrapper_path),
               "--canonical-sparse-bf16-state", str(args.canonical_sparse_bf16_state),
               "--canonical-sparse-nvfp4-state", str(args.canonical_sparse_nvfp4_state)]
    if args.force:
        command.append("--force")
    subprocess.run(command, check=True)
    exported = json.loads((args.output_dir / "phase_hetero_policy.json").read_text())
    if policy_sha256(exported) != policy_sha256(policy):
        raise RuntimeError("exported policy differs from requested policy")
    provenance = args.output_dir / "phase_hetero_export_provenance.json"
    if not provenance.is_file():
        raise RuntimeError("exporter did not write provenance")
    payload = json.loads(provenance.read_text())
    if payload.get("prune") is True:
        raise RuntimeError("runtime --prune is forbidden; use canonical states")


if __name__ == "__main__":
    main()

