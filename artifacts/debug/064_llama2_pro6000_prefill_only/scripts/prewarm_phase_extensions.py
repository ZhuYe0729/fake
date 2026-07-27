#!/usr/bin/env python3
"""Build/load phase-export CUDA extensions once before multi-process export."""
from __future__ import annotations

import sys

import torch


from common import CUTLASS


def sparse_2to4(rows: int, cols: int) -> torch.Tensor:
    weight = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    weight.reshape(-1, 4)[:, 2:] = 0
    return weight


def sparse_pairwise_4to8(rows: int, cols: int) -> torch.Tensor:
    weight = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    weight.reshape(-1, 4, 2)[:, 2:, :] = 0
    return weight


def main() -> None:
    sys.path.insert(0, str(CUTLASS))
    from cutlass_wrapper import (pack_sparse_bf16_weight, quantize_sparse_weight_bf16,
                                 quantize_weight_bf16)
    # Each call invokes the corresponding cached extension loader.  No model
    # checkpoint is involved; shapes only satisfy converter constraints.
    pack_sparse_bf16_weight(sparse_2to4(64, 64), prune=False)
    quantize_sparse_weight_bf16(sparse_pairwise_4to8(64, 64), prune=False)
    quantize_weight_bf16(torch.randn(32, 32, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    print("phase exporter extensions prewarmed")


if __name__ == "__main__":
    main()
