#!/usr/bin/env python3
"""Probe whether SparseNVFP4's pointer-keyed cache changes warm module latency.

This is a synthetic speed-only probe. It never supplies sparse weights to a
quality evaluation and is not a replacement for canonical SparseGPT weights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
sys.path.insert(0, str(CUTLASS))


def measure(modules, x, *, iterations: int, keep_outputs: bool):
    import torch

    torch.cuda.synchronize()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    retained = []
    with torch.no_grad():
        for index in range(iterations):
            out = modules[index % len(modules)](x)
            if keep_outputs:
                retained.append(out)
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    gpu_ms = start_event.elapsed_time(end_event)
    return {
        "gpu_ms": gpu_ms,
        "wall_ms": wall_ms,
        "calls": iterations,
        "gpu_ms_per_call": gpu_ms / iterations,
        "wall_ms_per_call": wall_ms / iterations,
        "retained_outputs": len(retained),
        "memory_allocated_bytes": torch.cuda.memory_allocated(),
        "memory_reserved_bytes": torch.cuda.memory_reserved(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    if args.tokens % 32:
        raise ValueError("tokens must be divisible by 32")

    import torch
    from cutlass_wrapper import SparseNVFP4Linear

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(17)
    x = torch.randn(args.tokens, args.in_features, device="cuda", dtype=torch.bfloat16)
    base = torch.nn.Linear(args.in_features, args.out_features, bias=False,
                           device="cuda", dtype=torch.bfloat16).eval()
    single = SparseNVFP4Linear.from_linear(base, prune=True).eval()
    layers = [single]
    for seed in range(1, args.layers):
        torch.manual_seed(17 + seed)
        linear = torch.nn.Linear(args.in_features, args.out_features, bias=False,
                                 device="cuda", dtype=torch.bfloat16).eval()
        layers.append(SparseNVFP4Linear.from_linear(linear, prune=True).eval())

    # Warm extension, allocator, and one-module cache state without retaining outputs.
    measure([single], x, iterations=4, keep_outputs=False)
    torch.cuda.empty_cache()
    result = {
        "shape": {"tokens": args.tokens, "n": args.out_features, "k": args.in_features},
        "layers": args.layers,
        "iterations": args.iterations,
        "single_reuse": measure([single], x, iterations=args.iterations, keep_outputs=False),
        "single_distinct_outputs": measure([single], x, iterations=args.iterations, keep_outputs=True),
        "layer_cycle_distinct_outputs": measure(layers, x, iterations=args.iterations, keep_outputs=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
