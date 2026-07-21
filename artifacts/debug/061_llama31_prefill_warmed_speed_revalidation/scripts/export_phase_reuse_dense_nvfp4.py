#!/usr/bin/env python3
"""Export phase policies while reusing the verified uniform dense-NVFP4 pack."""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPORTER = Path("/home/agent/wja/project/my/cospaq/test/vllm/artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py")
PACKED_DENSE_NVFP4 = ROOT / "artifacts/exports/vllm/baselines/llama3.1-8b-instruct/checkpoints/uniform_dense_nvfp4/model.safetensors"
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-sparse-bf16-state", type=Path)
    parser.add_argument("--canonical-sparse-nvfp4-state", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from safetensors import safe_open

    spec = importlib.util.spec_from_file_location("phase_export", EXPORTER)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {EXPORTER}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    packed = safe_open(str(PACKED_DENSE_NVFP4), framework="pt", device="cpu")

    def reuse_dense_nvfp4(name, _weight):
        keys = (f"{name}.weight", f"{name}.weight_scale", f"{name}.weight_global_scale")
        if any(key not in packed.keys() for key in keys): raise KeyError(f"missing dense-NVFP4 pack for {name}")
        return {key: packed.get_tensor(key).contiguous() for key in keys}

    module.quantize_dense_nvfp4 = reuse_dense_nvfp4
    argv = [str(EXPORTER), "--model-path", str(args.model_path), "--policy-json", str(args.policy_json),
            "--output-dir", str(args.output_dir), "--cutlass-wrapper-path", str(CUTLASS), "--force"]
    if args.canonical_sparse_bf16_state:
        argv += ["--canonical-sparse-bf16-state", str(args.canonical_sparse_bf16_state)]
    if args.canonical_sparse_nvfp4_state:
        argv += ["--canonical-sparse-nvfp4-state", str(args.canonical_sparse_nvfp4_state)]
    old = sys.argv; sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = old


if __name__ == "__main__": main()
