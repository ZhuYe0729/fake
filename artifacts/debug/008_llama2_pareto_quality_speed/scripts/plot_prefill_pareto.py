#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.use("Agg")

from common_pareto import DEBUG_ROOT, f, read_csv


COMPARISON_CSV = DEBUG_ROOT / "summary" / "prefill_only_comparison.csv"
PLOTS_DIR = DEBUG_ROOT / "plots"
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")
METHOD_COLORS = {
    "dense_bf16": "#4c72b0",
    "dense_nvfp4": "#55a868",
    "sparse_bf16": "#c44e52",
    "sparse_nvfp4": "#8172b2",
    "marlin_nvfp4": "#ccb974",
    "bf16": "#4c72b0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pareto plots for prefill_only comparison.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comp_path = args.output_root / "summary" / "prefill_only_comparison.csv"
    if not comp_path.exists():
        raise FileNotFoundError(f"missing {comp_path}; run build_baseline_comparison.py first")
    rows = read_csv(comp_path)
    pareto = [r for r in rows if r["row_type"] == "pareto"]
    uniform = [r for r in rows if r["row_type"] == "uniform"]
    pareto.sort(key=lambda r: f(r, "point_index"))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )

    plot_speed_vs_nll(pareto, uniform)
    plot_speed_vs_arc(pareto, uniform)
    plot_method_counts(pareto)
    plot_predicted_vs_e2e(pareto, uniform)
    print(f"wrote plots to {PLOTS_DIR}")


def plot_speed_vs_nll(pareto: list[dict], uniform: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    px = [f(r, "e2e_speedup_vs_dense") for r in pareto]
    py = [f(r, "nll_delta_vs_dense") for r in pareto]
    ax.plot(px, py, "o-", color="#2c3e50", linewidth=2, markersize=8, label="Pareto frontier", zorder=3)
    for i, r in enumerate(pareto):
        pi = int(f(r, "point_index"))
        if pi in (0, 5, 8, 10):
            ax.annotate(
                str(pi),
                (px[i], py[i]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=8,
                color="#2c3e50",
            )

    ux = [f(r, "e2e_speedup_vs_dense") for r in uniform]
    uy = [f(r, "nll_delta_vs_dense") for r in uniform]
    ulabels = [r["label"].replace("all_", "") for r in uniform]
    for i, r in enumerate(uniform):
        if ux[i] > 0 and uy[i] > 0:
            ax.scatter(ux[i], uy[i], marker="s", s=100, color="#e74c3c", zorder=4)
            ax.annotate(ulabels[i], (ux[i], uy[i]), textcoords="offset points", xytext=(6, -12), fontsize=7, color="#e74c3c")

    ax.scatter([], [], marker="s", s=80, color="#e74c3c", label="Uniform baselines")
    ax.set_xlabel("E2E speedup vs dense_bf16")
    ax.set_ylabel("NLL delta vs dense_bf16")
    ax.set_title("Llama2-7B prefill_only: Speed vs NLL trade-off")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"speed_vs_nll.{fmt}")
    plt.close(fig)


def plot_speed_vs_arc(pareto: list[dict], uniform: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    px = [f(r, "e2e_speedup_vs_dense") for r in pareto]
    py = [f(r, "arc_acc_norm") for r in pareto]
    ax.plot(px, py, "o-", color="#2c3e50", linewidth=2, markersize=8, label="Pareto frontier", zorder=3)
    for i, r in enumerate(pareto):
        pi = int(f(r, "point_index"))
        if pi in (0, 5, 8, 10):
            ax.annotate(
                str(pi),
                (px[i], py[i]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=8,
                color="#2c3e50",
            )

    ux = [f(r, "e2e_speedup_vs_dense") for r in uniform]
    uy = [f(r, "arc_acc_norm") for r in uniform]
    ulabels = [r["label"].replace("all_", "") for r in uniform]
    for i, r in enumerate(uniform):
        if ux[i] > 0 and uy[i] > 0:
            ax.scatter(ux[i], uy[i], marker="s", s=100, color="#e74c3c", zorder=4)
            ax.annotate(ulabels[i], (ux[i], uy[i]), textcoords="offset points", xytext=(6, -12), fontsize=7, color="#e74c3c")

    ax.scatter([], [], marker="s", s=80, color="#e74c3c", label="Uniform baselines")
    ax.set_xlabel("E2E speedup vs dense_bf16")
    ax.set_ylabel("ARC-Challenge acc_norm (limit=128)")
    ax.set_title("Llama2-7B prefill_only: Speed vs ARC-Challenge accuracy")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"speed_vs_arc_challenge.{fmt}")
    plt.close(fig)


def plot_method_counts(pareto: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    points = [int(f(r, "point_index")) for r in pareto]
    qcosts = [f(r, "quality_cost") for r in pareto]
    x_positions = list(range(len(pareto)))

    method_keys = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4"]
    counts: dict[str, list[float]] = {m: [0.0] * len(pareto) for m in method_keys}
    for i, r in enumerate(pareto):
        bc = parse_backend_counts(r.get("backend_counts", ""))
        for m in method_keys:
            counts[m][i] = float(bc.get(m, bc.get(m.replace("dense_", ""), 0)))

    bottom = [0.0] * len(pareto)
    width = 0.7
    for method in method_keys:
        color = METHOD_COLORS.get(method, "#888888")
        ax.bar(x_positions, counts[method], bottom=bottom, width=width, color=color, label=method, edgecolor="white", linewidth=0.5)
        bottom = [b + c for b, c in zip(bottom, counts[method])]

    for i, (xpos, bt, qc) in enumerate(zip(x_positions, bottom, qcosts)):
        ax.text(xpos, bt + 1, f"{points[i]}", ha="center", fontsize=8, color="#333333")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{qc:.1f}" for qc in qcosts], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Quality cost")
    ax.set_ylabel("Module count")
    ax.set_title("Llama2-7B prefill_only: Method counts along Pareto frontier")
    ax.legend(loc="center right")
    ax.set_ylim(0, 250)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"method_counts_frontier.{fmt}")
    plt.close(fig)


def plot_predicted_vs_e2e(pareto: list[dict], uniform: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    ppred = [f(r, "predicted_linear_latency_ms") for r in pareto]
    pe2e = [f(r, "e2e_prefill_mean_ms") for r in pareto]
    ax.scatter(ppred, pe2e, marker="o", s=60, color="#2c3e50", label="Pareto points", zorder=3)

    upred = [f(r, "predicted_linear_latency_ms") for r in uniform if f(r, "e2e_prefill_mean_ms") > 0]
    ue2e = [f(r, "e2e_prefill_mean_ms") for r in uniform if f(r, "e2e_prefill_mean_ms") > 0]
    ulabels = [r["label"].replace("all_", "") for r in uniform if f(r, "e2e_prefill_mean_ms") > 0]
    ax.scatter(upred, ue2e, marker="s", s=80, color="#e74c3c", label="Uniform baselines", zorder=4)
    for x, y, lbl in zip(upred, ue2e, ulabels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7, color="#e74c3c")

    all_pred = ppred + upred
    all_e2e = pe2e + ue2e
    if len(all_pred) >= 3:
        pr = pearson(all_pred, all_e2e)
        sp = spearman(all_pred, all_e2e)
        ax.text(
            0.05, 0.95, f"Pearson r = {pr:.4f}\nSpearman r = {sp:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    mn = min(all_pred + all_e2e) * 0.95
    mx = max(all_pred + all_e2e) * 1.02
    ax.plot([mn, mx], [mn, mx], "--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel("Predicted linear latency (ms)")
    ax.set_ylabel("E2E prefill latency (ms)")
    ax.set_title("Llama2-7B prefill_only: Predicted vs real E2E latency")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"predicted_vs_e2e_latency.{fmt}")
    plt.close(fig)


def parse_backend_counts(raw: str) -> dict[str, int]:
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return {str(k): int(v) for k, v in parsed.items()}
    except (ValueError, SyntaxError):
        pass
    try:
        parsed = json.loads(raw.replace("'", '"'))
        if isinstance(parsed, dict):
            return {str(k): int(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


if __name__ == "__main__":
    main()
