#!/usr/bin/env python3
"""Profile exact Llama3 prefill fused shapes without updating kernel models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
EXP = ROOT / "artifacts/debug/059_llama31_prefill_speed_decomposition"
sys.path[:0] = [str(CUTLASS), str(CUTLASS / "modeling")]

from modeling.kernel_predictor import KernelLatencyPredictor  # noqa: E402

SHAPES = (
    (16384, 6144, 4096),   # GQA qkv
    (16384, 4096, 4096),   # attention output
    (16384, 28672, 4096),  # fused gate/up
    (16384, 4096, 14336),  # MLP down
)
KERNELS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    args = parser.parse_args()
    output = EXP / "exact_micro"
    csv_path = output / "targeted_profile.csv"
    if csv_path.exists():
        print(csv_path)
        return
    predictor = KernelLatencyPredictor()
    result = predictor.profile(SHAPES, kernels=KERNELS, gpu=args.gpu,
                               warmup=args.warmup, iters=args.iters,
                               output_dir=output)
    print(result.output_csv)


if __name__ == "__main__":
    main()
