#!/usr/bin/env python3
"""Minimal GPU diagnostic for dense-NVFP4 packing used by checkpoint export."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def main() -> None:
    sys.path.insert(0, str(CUTLASS))
    import torch
    from cutlass_wrapper import quantize_weight_bf16
    torch.manual_seed(0)
    weight = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize(); start = time.perf_counter()
    packed = quantize_weight_bf16(weight, original_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(json.dumps({"elapsed_ms": (time.perf_counter() - start) * 1000.0,
                      "packed_shape": list(packed.packed_weight.shape),
                      "scale_shape": list(packed.scale.shape)}))


if __name__ == "__main__": main()
