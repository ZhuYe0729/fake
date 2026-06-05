#!/usr/bin/env python3
"""Pure prefill hybrid benchmark: batch_size=16, input_tokens=1024 (M=16384).

For each Qwen3.5 model (4B/9B/27B), reads existing module-level kernel
benchmark data at M=16384, determines the optimal kernel per linear layer,
and computes a comparison across all 6 methods (5 single + hybrid).

The hybrid method selects the fastest kernel independently for each
linear_group at M=16384 — no decode phase to worry about.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "artifacts/results/benchmarks/hybrid/prefill_only"

MODELS = ["4B", "9B", "27B"]
KERNELS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]
M_PREFILL = 16384  # batch_size=16 × input_tokens=1024


def _csv_path(model: str) -> Path:
    short = model.lower()
    return REPO_ROOT / f"artifacts/results/benchmarks/module/Qwen3.5-{model}/kernel/qwen35_{short}_module_kernel_curves.csv"


def load_kernel_data(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(row["m"]) == M_PREFILL and row["status"] == "pass":
                row["latency_ms"] = float(row["latency_ms"])
                row["m_val"] = int(row["m"])
                row["n_val"] = int(row["n"])
                row["k_val"] = int(row["k"])
                row["count"] = int(row["linear_count"])
                rows.append(row)
    return rows


def build_latency_map(rows: list[dict]) -> dict[tuple[str, str], float]:
    """{(linear_group, kernel): latency_ms}"""
    return {(r["linear_group"], r["kernel"]): r["latency_ms"] for r in rows}


def build_optimal_policy(rows: list[dict]) -> dict[str, tuple[str, float]]:
    """{linear_group: (best_kernel, latency_ms)}"""
    best: dict[str, tuple[str, float]] = {}
    for r in rows:
        g, k, lat = r["linear_group"], r["kernel"], r["latency_ms"]
        if g not in best or lat < best[g][1]:
            best[g] = (k, lat)
    return best


def method_total(lat_map: dict[tuple[str, str], float],
                 groups: list[str], counts: dict[str, int],
                 method: str) -> tuple[float, int]:
    """Total latency for a single-kernel method. Returns (total_ms, fallback_count)."""
    total = 0.0
    fallbacks = 0
    for g in groups:
        key = (g, method)
        if key in lat_map:
            total += counts[g] * lat_map[key]
        else:
            # Fallback to dense_bf16
            fb_key = (g, "dense_bf16")
            total += counts[g] * lat_map[fb_key]
            fallbacks += counts[g]
    return total, fallbacks


def hybrid_total(policy: dict[str, tuple[str, float]], counts: dict[str, int],
                 groups: list[str]) -> tuple[float, dict[str, int]]:
    """Hybrid total = sum of best per layer. Returns (total_ms, kernel_layer_counts)."""
    total = 0.0
    kernel_counts: dict[str, int] = {}
    for g in groups:
        k, lat = policy[g]
        total += counts[g] * lat
        kernel_counts[k] = kernel_counts.get(k, 0) + counts[g]
    return total, kernel_counts


def compute_model_results(model: str) -> dict:
    csv_path = _csv_path(model)
    rows = load_kernel_data(csv_path)
    lat_map = build_latency_map(rows)
    policy = build_optimal_policy(rows)

    groups_order = sorted(set(r["linear_group"] for r in rows))
    counts = {}
    shapes = {}
    for r in rows:
        g = r["linear_group"]
        if g not in counts:
            counts[g] = r["count"]
            shapes[g] = (r["n_val"], r["k_val"])

    # Base dense_bf16 total for speedup calculation
    dense_total, _ = method_total(lat_map, groups_order, counts, "dense_bf16")

    # Compute all method totals
    methods = {}
    for kernel in KERNELS:
        total, fb = method_total(lat_map, groups_order, counts, kernel)
        methods[kernel] = {
            "total_ms": total,
            "speedup": dense_total / total if total > 0 else 0,
            "fallbacks": fb,
        }

    hyb_total, hyb_kernel_counts = hybrid_total(policy, counts, groups_order)
    methods["hybrid"] = {
        "total_ms": hyb_total,
        "speedup": dense_total / hyb_total if hyb_total > 0 else 0,
        "fallbacks": 0,
        "kernel_counts": hyb_kernel_counts,
    }

    return {
        "model": f"Qwen3.5-{model}",
        "scenario": "prefill_only",
        "batch_size": 16,
        "input_tokens": 1024,
        "m_prefill": M_PREFILL,
        "dense_total_ms": dense_total,
        "groups_order": groups_order,
        "counts": counts,
        "shapes": shapes,
        "policy": policy,
        "lat_map": lat_map,
        "methods": methods,
    }


def print_model_report(result: dict) -> None:
    model = result["model"]
    policy = result["policy"]
    counts = result["counts"]
    groups = result["groups_order"]
    methods = result["methods"]

    print(f"\n{'='*90}")
    print(f"  {model} — Pure Prefill Hybrid (batch=16, in=1024, M={M_PREFILL})")
    print(f"{'='*90}")

    # Hybrid strategy details
    print(f"\n  Hybrid strategy (per-layer optimal at M={M_PREFILL}):")
    for g in groups:
        best_k, best_lat = policy[g]
        cnt = counts[g]
        n, k = result["shapes"][g]
        print(f"    {g:35s} N={n:5d} K={k:5d} → {best_k:15s} {best_lat:8.4f}ms ×{cnt:2d} = {best_lat*cnt:.3f}ms")

    kernels_used = set(policy[g][0] for g in groups)
    hyb_total = methods["hybrid"]["total_ms"]
    print(f"\n  Kernels used: {kernels_used}")
    print(f"  Hybrid prefill total: {hyb_total:.2f}ms")

    # Comparison table
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
    hyb = methods["hybrid"]
    print(f"  {'hybrid':<18s} {hyb['total_ms']:12.2f} {hyb['speedup']:9.4f}x  {'-':>5s}")

    print(f"\n  dense_bf16  = 1.0000x (baseline, {dense_total:.2f}ms)")
    print(f"  best single = {best_single_sp:.4f}x ({best_single_name})")
    print(f"  hybrid      = {hyb['speedup']:.4f}x")

    if hyb["speedup"] > best_single_sp:
        advantage = hyb["speedup"] / best_single_sp
        print(f"  hybrid / best_single = {advantage:.4f}x advantage")
    else:
        print(f"  hybrid ≈ best single (within margin)")


def write_results_csv(all_results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-model detailed CSVs
    for r in all_results:
        model_short = r["model"].replace("Qwen3.5-", "")
        path = OUTPUT_DIR / f"prefill_hybrid_{model_short.lower()}.csv"
        methods = r["methods"]
        dense_total = r["dense_total_ms"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "scenario", "batch_size", "input_tokens", "m_prefill",
                             "method", "prefill_ms", "speedup_vs_dense_bf16", "notes"])
            for kernel in KERNELS:
                m = methods[kernel]
                writer.writerow([r["model"], r["scenario"], r["batch_size"],
                                 r["input_tokens"], M_PREFILL,
                                 kernel, f"{m['total_ms']:.4f}",
                                 f"{m['speedup']:.4f}",
                                 f"fallbacks={m['fallbacks']}" if m['fallbacks'] else ""])
            hyb = methods["hybrid"]
            kernels_used = ",".join(f"{k}:{v}" for k, v in hyb.get("kernel_counts", {}).items())
            writer.writerow([r["model"], r["scenario"], r["batch_size"],
                             r["input_tokens"], M_PREFILL,
                             "hybrid", f"{hyb['total_ms']:.4f}",
                             f"{hyb['speedup']:.4f}",
                             f"kernels={kernels_used}"])
        print(f"  Wrote {path}")

    # Combined CSV
    combined_path = OUTPUT_DIR / "prefill_hybrid_all_models.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scenario", "batch_size", "input_tokens", "m_prefill",
                         "method", "prefill_ms", "speedup_vs_dense_bf16", "notes"])
        for r in all_results:
            methods = r["methods"]
            for kernel in KERNELS:
                m = methods[kernel]
                writer.writerow([r["model"], r["scenario"], r["batch_size"],
                                 r["input_tokens"], M_PREFILL,
                                 kernel, f"{m['total_ms']:.4f}",
                                 f"{m['speedup']:.4f}",
                                 f"fallbacks={m['fallbacks']}" if m['fallbacks'] else ""])
            hyb = methods["hybrid"]
            kernels_used = ",".join(f"{k}:{v}" for k, v in hyb.get("kernel_counts", {}).items())
            writer.writerow([r["model"], r["scenario"], r["batch_size"],
                             r["input_tokens"], M_PREFILL,
                             "hybrid", f"{hyb['total_ms']:.4f}",
                             f"{hyb['speedup']:.4f}",
                             f"kernels={kernels_used}"])
    print(f"  Wrote {combined_path}")


def write_summary_md(all_results: list[dict]) -> None:
    path = OUTPUT_DIR / "prefill_only_hybrid_summary.md"
    lines = []
    lines.append("# Qwen3.5 Pure Prefill Hybrid Benchmark")
    lines.append("")
    lines.append("## Scenario")
    lines.append("")
    lines.append("**batch_size=16, input_tokens=1024, output_tokens=1 (pure prefill)**")
    lines.append("")
    lines.append(f"- Prefill M = batch_size × input_tokens = {M_PREFILL}")
    lines.append("- Pure prefill means the prefill phase dominates E2E time")
    lines.append("- Benchmark uses module-level kernel latency data at M=16384")
    lines.append("- Baseline: dense_bf16 = 1.00x, higher = faster")
    lines.append("")
    lines.append("---")
    lines.append("")

    for r in all_results:
        model = r["model"]
        methods = r["methods"]
        dense_total = r["dense_total_ms"]
        policy = r["policy"]
        groups = r["groups_order"]
        counts = r["counts"]

        lines.append(f"## {model}")
        lines.append("")

        # Method comparison table
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
        kernels_str = ", ".join(f"{k}({v})" for k, v in hyb.get("kernel_counts", {}).items())
        lines.append(f"| **hybrid** | **{hyb['total_ms']:.2f}** | **{hyb['speedup']:.4f}x** | {kernels_str} |")
        lines.append("")

        advantage = hyb["speedup"] / best_single_sp if best_single_sp > 0 else 0
        lines.append(f"- Hybrid vs dense_bf16: **{hyb['speedup']:.4f}x**")
        lines.append(f"- Hybrid vs best single ({best_single_name} @ {best_single_sp:.4f}x): **{advantage:.4f}x**")
        lines.append("")

        # Per-layer strategy
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

        lines.append("---")
        lines.append("")

    # Combined table
    lines.append("## Combined Result Table")
    lines.append("")
    lines.append("| Method | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-27B |")
    lines.append("|--------|-----------|-----------|------------|")

    # Collect speedup per model per method
    speedups: dict[str, dict[str, float]] = {}
    for r in all_results:
        model = r["model"]
        speedups[model] = {}
        for kernel in KERNELS:
            speedups[model][kernel] = r["methods"][kernel]["speedup"]
        speedups[model]["hybrid"] = r["methods"]["hybrid"]["speedup"]

    all_methods = KERNELS + ["hybrid"]
    for method in all_methods:
        cells = [method]
        for r in all_results:
            model = r["model"]
            sp = speedups[model][method]
            total = r["methods"][method]["total_ms"]
            cells.append(f"{total:.2f}ms ({sp:.4f}x)")
        lines.append(f"| {' | '.join(cells)} |")

    lines.append("")
    lines.append(f"*Generated: 2026-06-03 | GPU: NVIDIA RTX 5090 32GB | PyTorch: 2.9.0 | CUDA: 12.8*")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def main():
    all_results = []
    for model in MODELS:
        print(f"\nProcessing Qwen3.5-{model}...")
        result = compute_model_results(model)
        all_results.append(result)
        print_model_report(result)

    print(f"\n{'='*90}")
    print("  Writing outputs...")
    write_results_csv(all_results)
    write_summary_md(all_results)
    print("  Done.")


if __name__ == "__main__":
    main()
