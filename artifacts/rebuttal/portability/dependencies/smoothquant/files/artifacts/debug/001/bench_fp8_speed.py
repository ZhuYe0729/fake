"""Benchmark FP8 E4M3 vs BF16 single Linear layer latency on RTX 5090.

Measures wall-clock GPU time for forward pass across diverse shapes and
batch sizes. Uses real FP8 matmul (torch._scaled_mm) — no fake quant.
"""

import csv
import os
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
from fp8_real_linear import Fp8RealLinear

# ── Config ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "fp8_speed_bench.csv")

WARMUP = 50
ITERS = 200

SHAPES = [
    # (in_features, out_features, label)
    (512, 512, "tiny-square"),
    (512, 2048, "tiny-expand"),
    (2048, 512, "tiny-contract"),
    (768, 3072, "bert-intermediate"),
    (3072, 768, "bert-output"),
    (4096, 4096, "llama-qkv"),
    (4096, 11008, "llama-up-gate"),
    (11008, 4096, "llama-down"),
    (8192, 8192, "large-square"),
    (4096, 16384, "large-expand"),
    (16384, 4096, "large-contract"),
]

BATCH_SIZES = [1, 4, 16, 64, 256, 1024, 2048, 4096]


def bench_forward(fn, x, warmup=WARMUP, iters=ITERS):
    """Time a forward pass using CUDA events. Returns median latency in ms."""
    # Warmup
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return torch.tensor(times).median().item()


def benchmark_one(in_features, out_features, batch_size):
    """Benchmark BF16 vs FP8 for a single (shape, batch_size) combination."""
    device = torch.device("cuda:0")

    # Create BF16 linear layer
    linear_bf16 = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)

    # Create FP8 linear layer from the same weights
    linear_fp8 = Fp8RealLinear.from_float(linear_bf16, weight_quant="per_channel", act_quant="per_token")

    # Random input
    x_bf16 = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)

    # Time BF16
    bf16_ms = bench_forward(lambda x: linear_bf16(x), x_bf16)

    # Time FP8 (Fp8RealLinear expects input in bfloat16, quantizes internally)
    fp8_ms = bench_forward(lambda x: linear_fp8(x), x_bf16)

    speedup = bf16_ms / fp8_ms if fp8_ms > 0 else 0.0

    return bf16_ms, fp8_ms, speedup


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}")
    print(f"Shapes: {len(SHAPES)}, Batch sizes: {len(BATCH_SIZES)}")
    print(f"Total combinations: {len(SHAPES) * len(BATCH_SIZES)}")
    print()
    print("=" * 90)

    header = f"{'Shape':<22s} {'(in,out)':<18s} {'M':>6s} {'BF16(ms)':>10s} {'FP8(ms)':>10s} {'Speedup':>8s}"
    print(header)
    print("-" * 90)

    results = []
    for in_f, out_f, label in SHAPES:
        for bs in BATCH_SIZES:
            try:
                bf16_ms, fp8_ms, speedup = benchmark_one(in_f, out_f, bs)
            except torch.cuda.OutOfMemoryError:
                print(f"{label:<22s} ({in_f},{out_f}) {bs:>6d}  OOM — skipping")
                results.append({
                    "shape": label, "in_features": in_f, "out_features": out_f,
                    "batch_size": bs, "M": bs,
                    "bf16_ms": None, "fp8_ms": None, "speedup": None,
                })
                torch.cuda.empty_cache()
                continue

            shape_str = f"({in_f},{out_f})"
            print(f"{label:<22s} {shape_str:<18s} {bs:>6d} {bf16_ms:>10.4f} {fp8_ms:>10.4f} {speedup:>7.2f}x")

            results.append({
                "shape": label, "in_features": in_f, "out_features": out_f,
                "batch_size": bs, "M": bs,
                "bf16_ms": round(bf16_ms, 4),
                "fp8_ms": round(fp8_ms, 4),
                "speedup": round(speedup, 4),
            })

        print("-" * 90)

    # Save CSV
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shape", "in_features", "out_features", "batch_size", "M", "bf16_ms", "fp8_ms", "speedup"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {RESULTS_FILE}")

    # Summary
    valid = [r["speedup"] for r in results if r["speedup"] is not None]
    if valid:
        print(f"\nSpeedup range: {min(valid):.2f}x — {max(valid):.2f}x, mean: {sum(valid)/len(valid):.2f}x")


if __name__ == "__main__":
    main()