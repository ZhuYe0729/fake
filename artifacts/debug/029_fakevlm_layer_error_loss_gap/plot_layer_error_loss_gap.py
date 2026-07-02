#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt


PROJECT_ROOT = ROOT.parents[2]
SOURCE_ROOT = PROJECT_ROOT / "artifacts" / "debug" / "024_fakevlm_prefill_global_pareto"
METHODS = ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
METHOD_LABELS = {
    "dense_nvfp4": "Dense NVFP4",
    "sparse_bf16": "Sparse BF16",
    "sparse_nvfp4": "Sparse NVFP4",
}
TYPE_COLORS = {
    "q_proj": "#4c78a8",
    "k_proj": "#72b7b2",
    "v_proj": "#f58518",
    "o_proj": "#54a24b",
    "gate_proj": "#b279a2",
    "up_proj": "#e45756",
    "down_proj": "#9d755d",
}


def main() -> None:
    cost_rows = read_csv(SOURCE_ROOT / "costs" / "batch_16" / "module_method_candidates.csv")
    loss_rows = read_csv(SOURCE_ROOT / "quality" / "stratified_loss.csv")
    quality_rows = read_csv(SOURCE_ROOT / "quality" / "stratified_quality.csv")
    cost_rows = [row for row in cost_rows if row["method"] in METHODS and row.get("supported", "True") == "True"]

    layer_summary = build_layer_summary(cost_rows)
    close_examples = find_close_error_examples(cost_rows, "sparse_nvfp4")
    policy_summary = build_policy_summary(loss_rows)
    accuracy_summary = build_accuracy_summary(quality_rows)
    method_stats = build_method_stats(cost_rows)

    write_csv(ROOT / "layer_method_summary.csv", layer_summary)
    write_csv(ROOT / "close_error_examples.csv", close_examples)
    write_csv(ROOT / "policy_loss_summary.csv", policy_summary)
    write_csv(ROOT / "policy_accuracy_summary.csv", accuracy_summary)
    write_json(ROOT / "summary.json", {"method_stats": method_stats, "top_close_error_pair": close_examples[:2]})

    plot_figure(cost_rows, layer_summary, policy_summary, accuracy_summary, close_examples)
    write_readme(method_stats, close_examples)
    print(f"wrote outputs to {ROOT}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def build_layer_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(f(row, "layer")))].append(row)
    out = []
    for (method, layer), items in sorted(grouped.items()):
        errors = [f(row, "output_rel_mse") for row in items]
        costs = [f(row, "quality_cost") for row in items]
        out.append(
            {
                "method": method,
                "layer": layer,
                "modules": len(items),
                "mean_output_rel_mse": mean(errors),
                "median_output_rel_mse": median(errors),
                "mean_quality_cost": mean(costs),
                "median_quality_cost": median(costs),
                "sum_quality_cost": sum(costs),
            }
        )
    return out


def build_policy_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        label = row.get("label", "")
        method = ""
        ratio = None
        for candidate in METHODS:
            prefix = f"{candidate}_ratio_"
            if label.startswith(prefix):
                method = candidate
                ratio = float(label[len(prefix) :])
                break
        if not method:
            continue
        counts = ast.literal_eval(row.get("backend_counts", "{}"))
        out.append(
            {
                "method": method,
                "ratio": ratio,
                "policy_index": row.get("policy_index", ""),
                "nll_delta_vs_dense": f(row, "nll_delta_vs_dense"),
                "nll": f(row, "nll"),
                "replaced_linear_count": int(f(row, "replaced_linear_count")),
                "selected_count": int(counts.get(method, 0)),
            }
        )
    return sorted(out, key=lambda row: (row["method"], row["ratio"]))


def build_accuracy_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    dense_accuracy = None
    for row in rows:
        if row.get("label") == "dense":
            dense_accuracy = f(row, "global_accuracy")
            break
    if dense_accuracy is None:
        dense_accuracy = max(f(row, "global_accuracy") for row in rows)
    for row in rows:
        label = row.get("label", "")
        method = ""
        ratio = None
        for candidate in METHODS:
            prefix = f"{candidate}_ratio_"
            if label.startswith(prefix):
                method = candidate
                ratio = float(label[len(prefix) :])
                break
        if not method:
            continue
        out.append(
            {
                "method": method,
                "ratio": ratio,
                "policy_index": row.get("policy_index", ""),
                "global_accuracy": f(row, "global_accuracy"),
                "accuracy_drop_vs_dense": dense_accuracy - f(row, "global_accuracy"),
                "total_wrong": int(f(row, "total_wrong")),
                "replaced_linear_count": int(f(row, "replaced_linear_count")),
            }
        )
    return sorted(out, key=lambda row: (row["method"], row["ratio"]))


def build_method_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out = {}
    for method in METHODS:
        items = [row for row in rows if row["method"] == method]
        errors = [f(row, "output_rel_mse") for row in items]
        costs = [f(row, "quality_cost") for row in items]
        layer_errors = []
        layer_costs = []
        for layer in sorted({int(f(row, "layer")) for row in items}):
            layer_items = [row for row in items if int(f(row, "layer")) == layer]
            layer_errors.append(mean(f(row, "output_rel_mse") for row in layer_items))
            layer_costs.append(mean(f(row, "quality_cost") for row in layer_items))
        out[method] = {
            "modules": len(items),
            "median_output_rel_mse": median(errors),
            "max_output_rel_mse_over_median": max(errors) / median(errors),
            "median_quality_cost": median(costs),
            "max_quality_cost_over_median": max(costs) / median(costs) if median(costs) else math.inf,
            "layer_mean_error_cv": pstdev(layer_errors) / mean(layer_errors),
            "layer_mean_quality_cost_cv": pstdev(layer_costs) / mean(layer_costs),
        }
    return out


def find_close_error_examples(rows: list[dict[str, str]], method: str) -> list[dict[str, object]]:
    items = [row for row in rows if row["method"] == method and f(row, "quality_cost") > 0]
    pairs = []
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            le = f(left, "output_rel_mse")
            re = f(right, "output_rel_mse")
            if min(le, re) <= 0:
                continue
            error_ratio = max(le, re) / min(le, re)
            if error_ratio > 1.08:
                continue
            lc = f(left, "quality_cost")
            rc = f(right, "quality_cost")
            cost_ratio = max(lc, rc) / min(lc, rc)
            if cost_ratio < 5:
                continue
            pairs.append((cost_ratio, error_ratio, left, right))
    out = []
    for pair_id, (cost_ratio, error_ratio, left, right) in enumerate(sorted(pairs, reverse=True, key=lambda item: item[0])[:8], start=1):
        for row in (left, right):
            out.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    "module_index": int(f(row, "module_index")),
                    "module_name": row["module_name"],
                    "layer": int(f(row, "layer")),
                    "module_type": row["module_type"],
                    "output_rel_mse": f(row, "output_rel_mse"),
                    "quality_cost": f(row, "quality_cost"),
                    "pair_error_ratio": error_ratio,
                    "pair_quality_cost_ratio": cost_ratio,
                }
            )
    return out


def plot_figure(
    rows: list[dict[str, str]],
    layer_summary: list[dict[str, object]],
    policy_summary: list[dict[str, object]],
    accuracy_summary: list[dict[str, object]],
    close_examples: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.8), constrained_layout=True)
    plot_sparse_scatter(axes[0], rows, close_examples)
    plot_layer_profiles(axes[1], layer_summary)
    plot_policy_loss(axes[2], policy_summary)
    plot_policy_accuracy(axes[3], accuracy_summary)
    fig.suptitle("FakeVLM: similar local compression error can map to very different loss impact", fontsize=13)
    fig.savefig(ROOT / "fakevlm_layer_error_loss_gap.png", dpi=220)
    fig.savefig(ROOT / "fakevlm_layer_error_loss_gap.pdf")
    plt.close(fig)


def plot_sparse_scatter(ax: plt.Axes, rows: list[dict[str, str]], close_examples: list[dict[str, object]]) -> None:
    items = [row for row in rows if row["method"] == "sparse_nvfp4"]
    for typ in sorted({row["module_type"] for row in items}):
        typ_rows = [row for row in items if row["module_type"] == typ]
        ax.scatter(
            [f(row, "output_rel_mse") for row in typ_rows],
            [f(row, "quality_cost") for row in typ_rows],
            s=24,
            alpha=0.78,
            label=typ,
            color=TYPE_COLORS.get(typ),
            edgecolors="none",
        )
    first_pair = [row for row in close_examples if row["pair_id"] == 1]
    if len(first_pair) == 2:
        xs = [row["output_rel_mse"] for row in first_pair]
        ys = [row["quality_cost"] for row in first_pair]
        ax.plot(xs, ys, color="#111111", linewidth=1.1)
        ax.scatter(xs, ys, s=80, facecolors="none", edgecolors="#111111", linewidths=1.6)
        ax.annotate(
            f"local error within {first_pair[0]['pair_error_ratio']:.2f}x\nloss proxy differs {first_pair[0]['pair_quality_cost_ratio']:.1f}x",
            xy=(max(xs), max(ys)),
            xytext=(-72, -18),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#222222"},
        )
    ax.set_title("Same method, close local error")
    ax.set_xlabel("Local output relative MSE")
    ax.set_ylabel("Per-module NLL-delta proxy")
    ax.set_yscale("log")
    ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    ax.legend(ncol=2, fontsize=7.5, frameon=False, loc="lower right")


def plot_layer_profiles(ax: plt.Axes, layer_summary: list[dict[str, object]]) -> None:
    rows = [row for row in layer_summary if row["method"] == "sparse_nvfp4"]
    layers = [int(row["layer"]) for row in rows]
    err = [float(row["mean_output_rel_mse"]) for row in rows]
    cost = [float(row["mean_quality_cost"]) for row in rows]
    err_norm = [value / median(err) for value in err]
    cost_norm = [value / median(cost) for value in cost]
    ax.plot(layers, err_norm, marker="o", markersize=3.2, linewidth=1.6, color="#4c78a8", label="mean local error / median")
    ax.plot(layers, cost_norm, marker="s", markersize=3.2, linewidth=1.6, color="#e45756", label="mean NLL proxy / median")
    ax.axhline(1.0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_title("Layer sensitivity expands the gap")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Normalized layer mean")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8, frameon=False, loc="upper right")


def plot_policy_loss(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    for method in METHODS:
        items = [row for row in rows if row["method"] == method]
        ax.plot(
            [float(row["ratio"]) for row in items],
            [float(row["nll_delta_vs_dense"]) for row in items],
            marker="o",
            linewidth=1.7,
            markersize=4,
            label=METHOD_LABELS[method],
        )
    ax.axhline(0.0, color="#555555", linewidth=0.7)
    ax.set_title("Measured full-model loss context")
    ax.set_xlabel("Fraction of modules compressed")
    ax.set_ylabel("Measured NLL delta vs dense")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def plot_policy_accuracy(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    for method in METHODS:
        items = [row for row in rows if row["method"] == method]
        ax.plot(
            [float(row["ratio"]) for row in items],
            [float(row["accuracy_drop_vs_dense"]) * 100.0 for row in items],
            marker="o",
            linewidth=1.7,
            markersize=4,
            label=METHOD_LABELS[method],
        )
    ax.axhline(0.0, color="#555555", linewidth=0.7)
    ax.set_title("Measured accuracy-drop context")
    ax.set_xlabel("Fraction of modules compressed")
    ax.set_ylabel("Accuracy drop vs dense (pp)")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def write_readme(method_stats: dict[str, dict[str, float]], close_examples: list[dict[str, object]]) -> None:
    pair = [row for row in close_examples if row["pair_id"] == 1]
    pair_text = "No close-error pair found."
    if len(pair) == 2:
        a, b = pair
        pair_text = (
            f"For `sparse_nvfp4`, `{a['module_name']}` and `{b['module_name']}` have local "
            f"`output_rel_mse` within {a['pair_error_ratio']:.3f}x, but the fitted per-module "
            f"NLL-delta proxy differs by {a['pair_quality_cost_ratio']:.2f}x."
        )
    stats_lines = []
    for method in METHODS:
        stats = method_stats[method]
        stats_lines.append(
            f"- `{method}`: layer mean local-error CV={stats['layer_mean_error_cv']:.3f}, "
            f"layer mean NLL-proxy CV={stats['layer_mean_quality_cost_cv']:.3f}, "
            f"max module NLL-proxy / median={stats['max_quality_cost_over_median']:.2f}x."
        )
    readme = f"""# FakeVLM Layer Error vs Loss Gap

This directory visualizes an existing FakeVLM result from `../024_fakevlm_prefill_global_pareto`.

## Main output

- `fakevlm_layer_error_loss_gap.png`
- `fakevlm_layer_error_loss_gap.pdf`

## What it shows

{pair_text}

The right-hand loss-impact value is not a newly measured single-layer NLL. It is the per-module quality cost from the already fitted `024` quality model, which was trained against measured full-model NLL rows in `quality/stratified_loss.csv` using the valid `assistant_answer_token_nll_v2_active_prefix_aligned` definition.

## Method stats

{chr(10).join(stats_lines)}

## Supporting files

- `layer_method_summary.csv`: per-method, per-layer local error and fitted quality-cost summary.
- `close_error_examples.csv`: close-local-error module pairs with large quality-cost differences.
- `policy_loss_summary.csv`: measured full-model NLL deltas for stratified method-ratio policies.
- `policy_accuracy_summary.csv`: measured FakeClue accuracy deltas for the same stratified policies.
- `summary.json`: compact machine-readable summary.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
