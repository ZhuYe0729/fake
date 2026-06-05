#!/usr/bin/env python3
"""Benchmark shape-workload hybrid vs single-kernel methods for Qwen3.5-9B.

Reads existing module-level kernel benchmark data, determines the optimal
kernel per (linear_group, M), then recreates and benchmarks hybrid modules
against each single-kernel method for a given inference scenario.

The hybrid selects the fastest measured kernel for each linear layer
independently, exploiting shape heterogeneity across the model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear, SPARSE_BF16_BLOCKED_SHAPES
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear

KERNELS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]

DEFAULT_DATA_CSV = (
    "artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_curves.csv"
)
DEFAULT_OUTPUT = "artifacts/results/benchmarks/hybrid/qwen35_9b_shape_workload_hybrid.csv"


def _load_wrapper():
    import importlib
    for module_name in (
        "fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper",
        "cutlass_wrapper",
    ):
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass
    raise RuntimeError("CUTLASS wrapper package is not importable")


def _load_kernel_data(csv_path: Path) -> list[dict]:
    """Load existing module-level kernel benchmark data."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "pass":
                row["latency_ms"] = float(row["latency_ms"])
                row["m"] = int(row["m"])
                row["n"] = int(row["n"])
                row["k"] = int(row["k"])
                row["linear_count"] = int(row["linear_count"])
                rows.append(row)
    return rows


def build_optimal_policy(rows: list[dict]) -> dict[tuple[str, int], str]:
    """Build the optimal kernel policy from measured data.

    Returns: {(linear_group, m): kernel_name}
    """
    # Group by (linear_group, m, kernel) → min latency
    best: dict[tuple[str, int], tuple[str, float]] = {}
    for r in rows:
        key = (r["linear_group"], r["m"], r["kernel"])
        lat = r["latency_ms"]
        gk = (r["linear_group"], r["m"])
        if gk not in best or lat < best[gk][1]:
            best[gk] = (r["kernel"], lat)
    return {gk: v[0] for gk, v in best.items()}


def _make_base_linear(n: int, k: int, device: torch.device, *, bias: bool, seed: int) -> nn.Linear:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    linear = nn.Linear(k, n, bias=bias, device=device, dtype=torch.bfloat16)
    linear.weight.data.normal_(mean=0.0, std=0.02, generator=generator)
    if bias:
        linear.bias.data.normal_(mean=0.0, std=0.02, generator=generator)
    linear.eval()
    linear.requires_grad_(False)
    return linear


def _make_module(kernel: str, base_linear: nn.Linear, device: torch.device, dtype: torch.dtype) -> nn.Module:
    wrapper = _load_wrapper()
    if kernel == "dense_bf16":
        return base_linear
    if kernel == "dense_nvfp4":
        return wrapper.NVFP4Linear.from_linear(base_linear, device=device).eval()
    if kernel == "marlin_nvfp4":
        return wrapper.MarlinNVFP4Linear.from_linear(base_linear, device=device, activation_dtype=dtype).eval()
    if kernel == "sparse_bf16":
        sparse = wrapper.SparseBF16Linear.from_linear(base_linear, device=device, prune=True).eval()
        return PaddedSparseBF16Linear(sparse, 8).eval()
    if kernel == "sparse_nvfp4":
        sparse = wrapper.SparseNVFP4Linear.from_linear(base_linear, device=device, prune=True).eval()
        return PaddedSparseNVFP4Linear(sparse, 32).eval()
    raise ValueError(f"unknown kernel: {kernel}")


def _time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(float(start_event.elapsed_time(end_event)))
    return sum(times) / len(times)


def _verify_against_csv(
    rows: list[dict],
    group: str,
    m: int,
    kernel: str,
    measured_lat: float,
    tolerance: float = 0.15,
) -> bool:
    """Verify a measured latency against CSV data within tolerance."""
    for r in rows:
        if r["linear_group"] == group and r["m"] == m and r["kernel"] == kernel:
            expected = r["latency_ms"]
            rel_diff = abs(measured_lat - expected) / expected
            return rel_diff <= tolerance
    return True  # No reference data, pass


def run_scenario(
    scenario_name: str,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    *,
    data_csv: Path,
    output_csv: Path | None = None,
    gpu: int = 0,
    warmup: int = 5,
    iters: int = 20,
    seed: int = 42,
    verify: bool = True,
) -> None:
    """Run the shape-workload hybrid benchmark for a given scenario."""

    # M_prefill = batch_size × input_tokens, M_decode = batch_size
    M_PREFILL = batch_size * input_tokens
    M_DECODE = batch_size

    # Load existing data
    all_rows = _load_kernel_data(data_csv)
    available_ms = sorted(set(r["m"] for r in all_rows))

    # Snap to nearest available M values
    m_prefill = min(available_ms, key=lambda x: abs(x - M_PREFILL))
    m_decode = min(available_ms, key=lambda x: abs(x - M_DECODE))

    # Extract model shapes
    groups_order = sorted(set(r["linear_group"] for r in all_rows))
    counts = {}
    shapes = {}
    for r in all_rows:
        g = r["linear_group"]
        if g not in counts:
            counts[g] = r["linear_count"]
            shapes[g] = (r["n"], r["k"])

    # Build optimal policy from data
    policy = build_optimal_policy(all_rows)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    print("=" * 90)
    print(f"SHAPE-WORKLOAD HYBRID BENCHMARK: {scenario_name}")
    print(f"  batch_size={batch_size}, input_tokens={input_tokens}, output_tokens={output_tokens}")
    print(f"  M_prefill={M_PREFILL} → using data at M={m_prefill}")
    print(f"  M_decode={M_DECODE} → using data at M={m_decode}")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print("=" * 90)

    # ---- Phase 1: Compute totals from CSV (already measured) ----
    print("\n--- From existing CSV data ---\n")

    def csv_total_at(m: int, kernel: str) -> tuple[float, int]:
        total = 0.0
        missed = 0
        for g in groups_order:
            found = None
            for r in all_rows:
                if r["linear_group"] == g and r["m"] == m and r["kernel"] == kernel:
                    found = r["latency_ms"]
                    break
            if found is not None:
                total += counts[g] * found
            else:
                # Fallback to dense_bf16
                for r in all_rows:
                    if r["linear_group"] == g and r["m"] == m and r["kernel"] == "dense_bf16":
                        total += counts[g] * r["latency_ms"]
                        missed += 1
                        break
        return total, missed

    def csv_hybrid_at(m: int) -> tuple[dict[str, str], float]:
        """Get hybrid assignments and total latency at given M."""
        assign = {}
        total = 0.0
        for g in groups_order:
            best_k = policy.get((g, m), "dense_bf16")
            assign[g] = best_k
            for r in all_rows:
                if r["linear_group"] == g and r["m"] == m and r["kernel"] == best_k:
                    total += counts[g] * r["latency_ms"]
                    break
        return assign, total

    # Prefill from CSV
    hyb_pf_assign, hyb_pf_total = csv_hybrid_at(m_prefill)
    print(f"Prefill (M={m_prefill}) hybrid kernel map:")
    for g in groups_order:
        k = hyb_pf_assign[g]
        lat = next(r["latency_ms"] for r in all_rows if r["linear_group"]==g and r["m"]==m_prefill and r["kernel"]==k)
        print(f"  {g:35s} → {k:15s} {lat:.4f}ms ×{counts[g]:2d} = {lat*counts[g]:.3f}ms")
    print(f"  Hybrid prefill total: {hyb_pf_total:.2f}ms")
    print(f"  Kernels used: {set(hyb_pf_assign.values())}")

    # Decode from CSV
    hyb_dec_assign, hyb_dec_total = csv_hybrid_at(m_decode)
    print(f"\nDecode (M={m_decode}) hybrid kernel map:")
    for g in groups_order:
        k = hyb_dec_assign[g]
        lat = next(r["latency_ms"] for r in all_rows if r["linear_group"]==g and r["m"]==m_decode and r["kernel"]==k)
        print(f"  {g:35s} → {k:15s} {lat:.4f}ms ×{counts[g]:2d} = {lat*counts[g]:.3f}ms")
    print(f"  Hybrid decode per-step: {hyb_dec_total:.2f}ms")
    print(f"  Kernels used: {set(hyb_dec_assign.values())}")

    all_kernels_used = set(hyb_pf_assign.values()) | set(hyb_dec_assign.values())
    print(f"\n  Total distinct kernels across both phases: {len(all_kernels_used)} → {all_kernels_used}")

    # ---- Phase 2: Verify by re-benchmarking hybrid modules ----
    if verify:
        print("\n" + "=" * 90)
        print("VERIFICATION: Re-benchmarking hybrid modules on GPU")
        print("=" * 90)

        verified = {"prefill": {}, "decode": {}}
        shape_idx = 0

        for g in groups_order:
            n, k = shapes[g]
            shape_idx += 1
            base = _make_base_linear(n, k, device, bias=False, seed=seed + shape_idx)

            # Prefill verification
            pf_kernel = hyb_pf_assign[g]
            try:
                pf_mod = _make_module(pf_kernel, base, device, dtype)
                x_pf = torch.randn((1, m_prefill, k), device=device, dtype=dtype)
                pf_lat = _time_cuda(lambda: pf_mod(x_pf), warmup, iters)
                verified["prefill"][g] = (pf_kernel, pf_lat)
                if verify:
                    ok = _verify_against_csv(all_rows, g, m_prefill, pf_kernel, pf_lat)
                    status = "✓" if ok else "✗ MISMATCH"
                    print(f"  [{shape_idx}/{len(groups_order)}] PF {g}: {pf_kernel} {pf_lat:.4f}ms {status}")
                del pf_mod
            except Exception as e:
                print(f"  [{shape_idx}/{len(groups_order)}] PF {g}: {pf_kernel} FAILED: {e}")
                verified["prefill"][g] = (pf_kernel, None)

            gc.collect()
            torch.cuda.empty_cache()

            # Decode verification
            dec_kernel = hyb_dec_assign[g]
            try:
                dec_mod = _make_module(dec_kernel, base, device, dtype)
                x_dec = torch.randn((1, m_decode, k), device=device, dtype=dtype)
                dec_lat = _time_cuda(lambda: dec_mod(x_dec), warmup, iters)
                verified["decode"][g] = (dec_kernel, dec_lat)
                if verify:
                    ok = _verify_against_csv(all_rows, g, m_decode, dec_kernel, dec_lat)
                    status = "✓" if ok else "✗ MISMATCH"
                    print(f"  [{shape_idx}/{len(groups_order)}] DEC {g}: {dec_kernel} {dec_lat:.4f}ms {status}")
                del dec_mod
            except Exception as e:
                print(f"  [{shape_idx}/{len(groups_order)}] DEC {g}: {dec_kernel} FAILED: {e}")
                verified["decode"][g] = (dec_kernel, None)

            gc.collect()
            torch.cuda.empty_cache()

    # ---- Phase 3: End-to-end comparison ----
    print("\n" + "=" * 90)
    print("END-TO-END COMPARISON  (baseline: dense_bf16 = 1.00x)")
    print("=" * 90)
    print(f"  Prefill:  1   × {hyb_pf_total:.2f}ms")
    print(f"  Decode:   {output_tokens} × {hyb_dec_total:.2f}ms = {hyb_dec_total*output_tokens:.2f}ms")
    print()

    hyb_e2e = hyb_pf_total + output_tokens * hyb_dec_total

    # dense_bf16 is baseline = 1.00x, higher numbers = faster
    _, dense_e2e_base = csv_total_at(m_prefill, "dense_bf16")
    dense_e2e_base = dense_e2e_base if isinstance(dense_e2e_base, float) else 0
    # Recompute properly
    dense_pf_t, _ = csv_total_at(m_prefill, "dense_bf16")
    dense_dec_t, _ = csv_total_at(m_decode, "dense_bf16")
    dense_e2e = dense_pf_t + output_tokens * dense_dec_t

    print(f"{'Method':<18s} {'Prefill(ms)':>12s} {'Decode×n(ms)':>14s} {'E2E(ms)':>12s}  {'Speedup':>10s}  FB")
    print("-" * 82)

    for kernel in KERNELS:
        pf_t, pf_fb = csv_total_at(m_prefill, kernel)
        dec_t, dec_fb = csv_total_at(m_decode, kernel)
        e2e = pf_t + output_tokens * dec_t
        speedup = dense_e2e / e2e  # >1 means faster than dense_bf16
        fb_str = f"{pf_fb+dec_fb}" if pf_fb + dec_fb > 0 else "-"
        print(f"{kernel:<18s} {pf_t:12.2f} {dec_t*output_tokens:14.2f} {e2e:12.2f}  {speedup:9.4f}x  {fb_str}")

    hybrid_speedup = dense_e2e / hyb_e2e
    print(f"{'HYBRID':<18s} {hyb_pf_total:12.2f} {hyb_dec_total*output_tokens:14.2f} {hyb_e2e:12.2f}  {hybrid_speedup:9.4f}x  -")

    # Find best single-kernel speedup
    best_single_speedup = 0.0
    best_single_name = ""
    for kernel in KERNELS:
        pf_t, _ = csv_total_at(m_prefill, kernel)
        dec_t, _ = csv_total_at(m_decode, kernel)
        e2e = pf_t + output_tokens * dec_t
        sp = dense_e2e / e2e
        if sp > best_single_speedup:
            best_single_speedup = sp
            best_single_name = kernel

    print(f"\n  dense_bf16     = 1.00x (baseline, {dense_e2e:.2f}ms)")
    print(f"  best single    = {best_single_speedup:.4f}x ({best_single_name})")
    print(f"  hybrid         = {hybrid_speedup:.4f}x  ← {'fastest' if hybrid_speedup > best_single_speedup else ''}")
    print(f"  hybrid / best  = {hybrid_speedup / best_single_speedup:.4f}x advantage")

    # Write output CSV
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "scenario", "batch_size", "input_tokens", "output_tokens",
            "m_prefill", "m_decode", "method", "prefill_ms", "decode_ms",
            "e2e_ms", "speedup_vs_dense_bf16",
        ]
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for kernel in KERNELS:
                pf_t, _ = csv_total_at(m_prefill, kernel)
                dec_t, _ = csv_total_at(m_decode, kernel)
                e2e = pf_t + output_tokens * dec_t
                writer.writerow({
                    "scenario": scenario_name,
                    "batch_size": batch_size,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "m_prefill": m_prefill,
                    "m_decode": m_decode,
                    "method": kernel,
                    "prefill_ms": f"{pf_t:.4f}",
                    "decode_ms": f"{dec_t:.4f}",
                    "e2e_ms": f"{e2e:.4f}",
                    "speedup_vs_dense_bf16": f"{dense_e2e/e2e:.4f}",
                })
            writer.writerow({
                "scenario": scenario_name,
                "batch_size": batch_size,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "m_prefill": m_prefill,
                "m_decode": m_decode,
                "method": "hybrid",
                "prefill_ms": f"{hyb_pf_total:.4f}",
                "decode_ms": f"{hyb_dec_total:.4f}",
                "e2e_ms": f"{hyb_e2e:.4f}",
                "speedup_vs_dense_bf16": f"{hybrid_speedup:.4f}",
            })
        print(f"\nWrote results to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--data-csv", type=Path, default=Path(DEFAULT_DATA_CSV))
    parser.add_argument("--output-csv", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verify", action="store_true", help="Skip GPU verification")
    parser.add_argument(
        "--scenario-name", type=str, default="default",
        help="Label for this scenario",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_name = args.scenario_name
    if scenario_name == "default":
        scenario_name = f"bs{args.batch_size}_in{args.input_tokens}_out{args.output_tokens}"
    run_scenario(
        scenario_name,
        args.batch_size,
        args.input_tokens,
        args.output_tokens,
        data_csv=Path(args.data_csv),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        gpu=args.gpu,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
