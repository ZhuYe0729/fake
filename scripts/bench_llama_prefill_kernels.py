#!/usr/bin/env python3
"""Benchmark Llama linear kernels at M=16384 (pure prefill).

Extracts linear shapes from Llama-2-7B and Llama-3.1-8B safetensors,
benchmarks all 5 kernels at M=16384, outputs module-level CSV data,
then computes hybrid strategy and generates comparison tables.
"""

from __future__ import annotations

import csv
import gc
import re
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from safetensors import safe_open

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.cutlass_sparse_bf16 import (
    PaddedSparseBF16Linear,
    SPARSE_BF16_BLOCKED_SHAPES,
)
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear

KERNELS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]
M_PREFILL = 16384

LLAMA_LINEAR_SUFFIXES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

LLAMA_MODELS = {
    "Llama-2-7B": "/home/agent/wja/data/models/LLM-Research/llama-2-7b",
    "Llama-3.1-8B": "/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
}

Q35_9B_CSV = REPO_ROOT / "artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_curves.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts/results/benchmarks/hybrid/prefill_only"
KERNEL_CSV_DIR = OUTPUT_DIR / "kernel_data"


def _load_wrapper():
    import importlib
    for mn in ("fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper", "cutlass_wrapper"):
        try:
            return importlib.import_module(mn)
        except Exception:
            pass
    raise RuntimeError("CUTLASS wrapper package is not importable")


def extract_llama_shapes(model_path: str) -> list[dict]:
    """Extract unique linear shapes from Llama safetensors."""
    files = sorted(Path(model_path).glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensors in {model_path}")

    pattern = re.compile(r"^model\.layers\.\d+\.(.+)\.weight$")
    counts: dict[str, dict[tuple[int, int], int]] = {s: {} for s in LLAMA_LINEAR_SUFFIXES}

    for fp in files:
        with safe_open(fp, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                m = pattern.match(name)
                if not m:
                    continue
                suffix = m.group(1)
                if suffix not in counts:
                    continue
                shape = tuple(int(v) for v in tensors.get_slice(name).get_shape())
                if len(shape) != 2:
                    continue
                counts[suffix][shape] = counts[suffix].get(shape, 0) + 1

    shapes = []
    for suffix in LLAMA_LINEAR_SUFFIXES:
        gs = counts[suffix]
        if not gs:
            continue
        if len(gs) != 1:
            raise RuntimeError(f"Multiple shapes for {suffix}: {gs}")
        (n, k), count = next(iter(gs.items()))
        shapes.append({"group": suffix, "count": count, "n": n, "k": k})
    return shapes


def _make_base_linear(n: int, k: int, device: torch.device, seed: int) -> nn.Linear:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    lin = nn.Linear(k, n, bias=False, device=device, dtype=torch.bfloat16)
    lin.weight.data.normal_(mean=0.0, std=0.02, generator=g)
    lin.eval()
    lin.requires_grad_(False)
    return lin


def _kernel_supported(kernel: str, n: int, k: int) -> tuple[bool, str]:
    wrapper = _load_wrapper()
    if kernel == "dense_bf16":
        return True, ""
    if kernel == "dense_nvfp4":
        if not wrapper.can_use_cutlass_nvfp4(1, n, k, load_extension=False):
            return False, f"shape_not_supported:dense_nvfp4:N={n},K={k}"
        return True, ""
    if kernel == "marlin_nvfp4":
        if not wrapper.can_use_marlin_nvfp4(1, n, k, load_extension=False):
            return False, f"shape_not_supported:marlin_nvfp4:N={n},K={k}"
        return True, ""
    if kernel == "sparse_bf16":
        if (n, k) in SPARSE_BF16_BLOCKED_SHAPES:
            return False, f"shape_blocked:sparse_bf16:N={n},K={k}"
        if not wrapper.can_use_cutlass_sparse_bf16(n, 8, k, load_extension=False):
            return False, f"shape_not_supported:sparse_bf16:N={n},K={k}"
        return True, ""
    if kernel == "sparse_nvfp4":
        if not wrapper.can_use_cutlass_sparse_nvfp4(n, 32, k, load_extension=False):
            return False, f"shape_not_supported:sparse_nvfp4:N={n},K={k}"
        return True, ""
    raise ValueError(f"unknown kernel: {kernel}")


@torch.no_grad()
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
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return sum(times) / len(times)


def benchmark_llama_model(model_name: str, model_path: str, gpu: int = 0,
                          warmup: int = 5, iters: int = 20) -> list[dict]:
    """Benchmark all linear shapes for a Llama model at M=16384."""
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    shapes = extract_llama_shapes(model_path)
    rows = []

    print(f"\n{'='*70}")
    print(f"  {model_name} — Module Kernel Benchmarks at M={M_PREFILL}")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"{'='*70}")

    for si, shape in enumerate(shapes, 1):
        g, cnt, n, k = shape["group"], shape["count"], shape["n"], shape["k"]
        print(f"\n  [{si}/{len(shapes)}] {g}  N={n}, K={k}  ×{cnt} layers")

        base = _make_base_linear(n, k, device, seed=42 + si)

        for kernel in KERNELS:
            supported, reason = _kernel_supported(kernel, n, k)
            row = {
                "model": model_name,
                "benchmark_level": "module_forward",
                "linear_group": g,
                "linear_count": cnt,
                "m": M_PREFILL,
                "n": n,
                "k": k,
                "kernel": kernel,
                "gpu": gpu,
                "warmup": warmup,
                "iters": iters,
                "bias": False,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }

            if not supported:
                row["status"] = "skip"
                row["error_msg"] = reason
                row["latency_ms"] = ""
                print(f"    {kernel:15s} → SKIP ({reason})")
            else:
                try:
                    mod = _make_module(kernel, base, device, dtype)
                    x = torch.randn((1, M_PREFILL, k), device=device, dtype=dtype)
                    lat = _time_cuda(lambda: mod(x), warmup, iters)
                    row["status"] = "pass"
                    row["latency_ms"] = lat
                    tflops = (2.0 * M_PREFILL * n * k) / (lat * 1e-3) / 1e12
                    print(f"    {kernel:15s} → {lat:8.4f}ms  ({tflops:.1f} TFLOPS)")
                    del mod
                except Exception as exc:
                    msg = str(exc).replace("\n", " ")
                    if "out of memory" in msg.lower():
                        row["status"] = "skip"
                        row["error_msg"] = f"OOM:{msg[:200]}"
                    else:
                        row["status"] = "error"
                        row["error_msg"] = msg[:500]
                    row["latency_ms"] = ""
                    print(f"    {kernel:15s} → FAIL: {msg[:80]}")

            rows.append(row)
            gc.collect()
            torch.cuda.empty_cache()

        del base
        gc.collect()
        torch.cuda.empty_cache()

    return rows


def load_qwen35_9b_data() -> list[dict]:
    """Load existing Qwen3.5-9B kernel data at M=16384."""
    rows = []
    with open(Q35_9B_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["m"]) == M_PREFILL and row["status"] == "pass":
                row["latency_ms"] = float(row["latency_ms"])
                row["m_val"] = int(row["m"])
                row["n_val"] = int(row["n"])
                row["k_val"] = int(row["k"])
                row["count"] = int(row["linear_count"])
                rows.append(row)
    return rows


# ─── Analysis ───

def build_latency_map(rows: list[dict]) -> dict[tuple[str, str], float]:
    return {(r["linear_group"], r["kernel"]): r["latency_ms"] for r in rows}


def build_optimal_policy(rows: list[dict]) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for r in rows:
        g, k, lat = r["linear_group"], r["kernel"], r["latency_ms"]
        if g not in best or lat < best[g][1]:
            best[g] = (k, lat)
    return best


def method_total(lat_map: dict, groups: list[str], counts: dict[str, int],
                 method: str) -> tuple[float, int]:
    total = 0.0
    fallbacks = 0
    for g in groups:
        key = (g, method)
        if key in lat_map:
            total += counts[g] * lat_map[key]
        else:
            total += counts[g] * lat_map[(g, "dense_bf16")]
            fallbacks += counts[g]
    return total, fallbacks


def hybrid_total(policy: dict, counts: dict[str, int],
                 groups: list[str]) -> tuple[float, dict[str, int]]:
    total = 0.0
    kc: dict[str, int] = {}
    for g in groups:
        k, lat = policy[g]
        total += counts[g] * lat
        kc[k] = kc.get(k, 0) + counts[g]
    return total, kc


def compute_model_results(model_name: str, rows: list[dict]) -> dict:
    lat_map = build_latency_map(rows)
    policy = build_optimal_policy(rows)
    groups_order = sorted(set(r["linear_group"] for r in rows))
    counts = {}
    shapes = {}
    for r in rows:
        g = r["linear_group"]
        if g not in counts:
            counts[g] = r["count"] if "count" in r else r["linear_count"]
            shapes[g] = (r["n"] if "n" in r else r["n_val"],
                         r["k"] if "k" in r else r["k_val"])

    dense_total, _ = method_total(lat_map, groups_order, counts, "dense_bf16")

    methods = {}
    for kernel in KERNELS:
        total, fb = method_total(lat_map, groups_order, counts, kernel)
        methods[kernel] = {
            "total_ms": total,
            "speedup": dense_total / total if total > 0 else 0,
            "fallbacks": fb,
        }

    hyb_total, hyb_kc = hybrid_total(policy, counts, groups_order)
    methods["hybrid"] = {
        "total_ms": hyb_total,
        "speedup": dense_total / hyb_total if hyb_total > 0 else 0,
        "fallbacks": 0,
        "kernel_counts": hyb_kc,
    }

    return {
        "model": model_name,
        "dense_total_ms": dense_total,
        "groups_order": groups_order,
        "counts": counts,
        "shapes": shapes,
        "policy": policy,
        "methods": methods,
    }


def print_report(result: dict) -> None:
    model = result["model"]
    policy = result["policy"]
    counts = result["counts"]
    groups = result["groups_order"]
    methods = result["methods"]

    print(f"\n{'='*80}")
    print(f"  {model} — Pure Prefill Hybrid (M={M_PREFILL})")
    print(f"{'='*80}")

    print(f"\n  Hybrid per-layer strategy:")
    for g in groups:
        best_k, best_lat = policy[g]
        cnt = counts[g]
        n, k = result["shapes"][g]
        print(f"    {g:30s} N={n:5d} K={k:5d} → {best_k:15s} {best_lat:8.4f}ms ×{cnt:2d} = {best_lat*cnt:.3f}ms")

    kernels_used = set(policy[g][0] for g in groups)
    hyb = methods["hybrid"]
    print(f"\n  Kernels used: {kernels_used}")
    print(f"  Hybrid prefill total: {hyb['total_ms']:.2f}ms")

    dense_total = result["dense_total_ms"]
    print(f"\n  {'Method':<18s} {'Prefill(ms)':>12s} {'Speedup':>10s}  {'FB':>5s}")
    print(f"  {'-'*50}")

    best_single_sp = 0.0
    best_single_name = ""
    for kernel in KERNELS:
        m = methods[kernel]
        fb = m["fallbacks"]
        fb_str = str(fb) if fb > 0 else "-"
        print(f"  {kernel:<18s} {m['total_ms']:12.2f} {m['speedup']:9.4f}x  {fb_str:>5s}")
        if m["speedup"] > best_single_sp:
            best_single_sp = m["speedup"]
            best_single_name = kernel

    print(f"  {'─'*50}")
    print(f"  {'hybrid':<18s} {hyb['total_ms']:12.2f} {hyb['speedup']:9.4f}x  {'-':>5s}")
    print(f"\n  dense_bf16  = 1.0000x (baseline, {dense_total:.2f}ms)")
    print(f"  best single = {best_single_sp:.4f}x ({best_single_name})")
    print(f"  hybrid      = {hyb['speedup']:.4f}x")
    if hyb["speedup"] > best_single_sp:
        print(f"  hybrid / best_single = {hyb['speedup'] / best_single_sp:.4f}x")


def write_outputs(all_results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Per-model CSVs
    for r in all_results:
        model_short = r["model"].lower().replace(".", "").replace("-", "_")
        path = OUTPUT_DIR / f"prefill_hybrid_{model_short}.csv"
        methods = r["methods"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "scenario", "batch_size", "input_tokens", "m_prefill",
                         "method", "prefill_ms", "speedup_vs_dense_bf16", "notes"])
            for kernel in KERNELS:
                m = methods[kernel]
                w.writerow([r["model"], "prefill_only", 16, 1024, M_PREFILL,
                            kernel, f"{m['total_ms']:.4f}", f"{m['speedup']:.4f}",
                            f"fallbacks={m['fallbacks']}" if m['fallbacks'] else ""])
            hyb = methods["hybrid"]
            kc_str = ",".join(f"{k}:{v}" for k, v in hyb["kernel_counts"].items())
            w.writerow([r["model"], "prefill_only", 16, 1024, M_PREFILL,
                        "hybrid", f"{hyb['total_ms']:.4f}", f"{hyb['speedup']:.4f}",
                        f"kernels={kc_str}"])
        print(f"  Wrote {path}")

    # Combined CSV
    combined_path = OUTPUT_DIR / "prefill_hybrid_all_models.csv"
    with open(combined_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "scenario", "batch_size", "input_tokens", "m_prefill",
                     "method", "prefill_ms", "speedup_vs_dense_bf16", "notes"])
        for r in all_results:
            methods = r["methods"]
            for kernel in KERNELS:
                m = methods[kernel]
                w.writerow([r["model"], "prefill_only", 16, 1024, M_PREFILL,
                            kernel, f"{m['total_ms']:.4f}", f"{m['speedup']:.4f}",
                            f"fallbacks={m['fallbacks']}" if m['fallbacks'] else ""])
            hyb = methods["hybrid"]
            kc_str = ",".join(f"{k}:{v}" for k, v in hyb["kernel_counts"].items())
            w.writerow([r["model"], "prefill_only", 16, 1024, M_PREFILL,
                        "hybrid", f"{hyb['total_ms']:.4f}", f"{hyb['speedup']:.4f}",
                        f"kernels={kc_str}"])
    print(f"  Wrote {combined_path}")

    # Summary markdown
    md_path = OUTPUT_DIR / "prefill_only_hybrid_summary.md"
    lines = [
        "# Pure Prefill Hybrid Benchmark — Llama & Qwen3.5",
        "",
        "## Scenario",
        "",
        "**batch_size=16, input_tokens=1024, output_tokens=1 (pure prefill)**",
        "",
        f"- Prefill M = batch_size × input_tokens = {M_PREFILL}",
        "- Hybrid = each linear layer independently selects the fastest kernel at M=16384",
        "- Baseline: dense_bf16 = 1.00x, higher = faster",
        "- Data source: module-level kernel benchmarks at M=16384",
        "",
        "---",
        "",
        "## Combined Result Table",
        "",
        "| Method | Llama-2-7B | Llama-3.1-8B | Qwen3.5-9B |",
        "|--------|-----------|-------------|-----------|",
    ]

    # Reorder: Llama-2-7B, Llama-3.1-8B, Qwen3.5-9B
    ordered = sorted(all_results, key=lambda r: ["Llama-2-7B", "Llama-3.1-8B", "Qwen3.5-9B"].index(r["model"]))

    all_methods = KERNELS + ["hybrid"]
    for method in all_methods:
        cells = [method]
        for r in ordered:
            m = r["methods"][method]
            cells.append(f"{m['total_ms']:.2f}ms ({m['speedup']:.4f}x)")
        lines.append(f"| {' | '.join(cells)} |")

    lines.append("")

    # Per-model detail sections
    for r in ordered:
        model = r["model"]
        methods = r["methods"]
        policy = r["policy"]
        groups = r["groups_order"]
        counts = r["counts"]
        dense_total = r["dense_total_ms"]

        lines.append("---")
        lines.append(f"## {model}")
        lines.append("")

        lines.append("| Method | Prefill(ms) | Speedup | Notes |")
        lines.append("|--------|------------|---------|-------|")

        best_single_sp = 0.0
        best_single_name = ""
        for kernel in KERNELS:
            m = methods[kernel]
            fb = m["fallbacks"]
            note = f"{fb} layers fallback to dense_bf16" if fb > 0 else ""
            lines.append(f"| {kernel} | {m['total_ms']:.2f} | {m['speedup']:.4f}x | {note} |")
            if m["speedup"] > best_single_sp:
                best_single_sp = m["speedup"]
                best_single_name = kernel

        hyb = methods["hybrid"]
        kc_str = ", ".join(f"{k}({v})" for k, v in hyb["kernel_counts"].items())
        lines.append(f"| **hybrid** | **{hyb['total_ms']:.2f}** | **{hyb['speedup']:.4f}x** | {kc_str} |")
        lines.append("")

        advantage = hyb["speedup"] / best_single_sp if best_single_sp > 0 else 0
        lines.append(f"- Hybrid vs dense_bf16: **{hyb['speedup']:.4f}x**")
        lines.append(f"- Hybrid vs best single ({best_single_name} @ {best_single_sp:.4f}x): **{advantage:.4f}x**")
        lines.append("")

        lines.append("### Hybrid Strategy (M=16384)")
        lines.append("")
        lines.append("| Layer | N | K | Best Kernel | Latency(ms) | Count | Subtotal(ms) |")
        lines.append("|-------|---|---|------------|------------|-------|-------------|")
        for g in groups:
            best_k, best_lat = policy[g]
            cnt = counts[g]
            n, k = r["shapes"][g]
            lines.append(f"| {g} | {n} | {k} | {best_k} | {best_lat:.4f} | {cnt} | {best_lat*cnt:.3f} |")
        lines.append("")

    lines.append(f"*Generated: 2026-06-03 | GPU: NVIDIA RTX 5090 32GB | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}*")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {md_path}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    all_results = []

    # Benchmark Llama models
    for model_name, model_path in LLAMA_MODELS.items():
        rows = benchmark_llama_model(model_name, model_path)

        # Save kernel CSV
        csv_path = KERNEL_CSV_DIR / f"{model_name.lower().replace('-', '_')}_kernel_m16384.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "model", "benchmark_level", "linear_group", "linear_count",
                "m", "n", "k", "kernel", "status", "latency_ms",
                "gpu", "warmup", "iters", "bias", "torch_version",
                "cuda_version", "error_msg",
            ])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        print(f"  Kernel data saved to {csv_path}")

        # Filter to passing rows
        pass_rows = [r for r in rows if r["status"] == "pass" and r.get("latency_ms")]
        result = compute_model_results(model_name, pass_rows)
        all_results.append(result)
        print_report(result)

    # Load Qwen3.5-9B data
    print(f"\n{'='*70}")
    print(f"  Loading Qwen3.5-9B data from existing CSV...")
    q35_rows = load_qwen35_9b_data()
    # Adapt column names
    for r in q35_rows:
        r["n"] = r.pop("n_val")
        r["k"] = r.pop("k_val")
        r["count"] = r.pop("count")
        r["m"] = r.pop("m_val")
    q35_result = compute_model_results("Qwen3.5-9B", q35_rows)
    all_results.append(q35_result)
    print_report(q35_result)

    # Write outputs
    print(f"\n{'='*70}")
    print(f"  Writing outputs to {OUTPUT_DIR}...")
    write_outputs(all_results)
    print(f"  Done!")


if __name__ == "__main__":
    main()
