#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from common_prefill_loss import DEBUG_ROOT, f, read_csv, write_csv, write_json


LOCAL_METRICS = (
    "weight_mse",
    "weight_rel_mse",
    "weight_rmse_over_rms",
    "weight_max_abs_error",
    "output_mse",
    "output_rel_mse",
    "output_rmse_over_rms",
    "output_max_abs_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prefill-only local error and loss ablation results.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir or args.output_root / "summary" / "prefill_loss_modeling"
    summary_dir.mkdir(parents=True, exist_ok=True)
    local_rows = read_csv(args.output_root / "sensitivity" / "module_method_local_errors.csv")
    ablation_rows = load_ablation_rows(args.output_root)
    layer_type = [row for row in ablation_rows if row.get("policy_kind") == "layer_type"]
    type_rows = [row for row in ablation_rows if row.get("policy_kind") == "type"]
    layer_rows = [row for row in ablation_rows if row.get("policy_kind") == "layer"]

    joined = join_layer_type(local_rows, layer_type)
    correlations = compute_correlations(joined)
    layer_summary = summarize_group(layer_rows, ["method", "layer"])
    type_summary = summarize_group(type_rows, ["method", "linear_type"])
    method_summary = summarize_group([row for row in ablation_rows if row.get("method") != "dense_bf16"], ["method"])

    write_csv(summary_dir / "layer_type_local_error_loss_joined.csv", joined)
    write_csv(summary_dir / "local_error_loss_correlations.csv", correlations)
    write_csv(summary_dir / "layer_loss_summary.csv", layer_summary)
    write_csv(summary_dir / "linear_type_loss_summary.csv", type_summary)
    write_csv(summary_dir / "method_loss_summary.csv", method_summary)
    plot_layer_depth(layer_summary, summary_dir / "layer_depth_loss_delta.png")
    plot_type_loss(type_summary, summary_dir / "linear_type_loss_delta.png")
    plot_best_proxy(joined, correlations, summary_dir / "best_local_proxy_vs_loss_delta.png")
    plot_weight_vs_output(local_rows, summary_dir / "weight_rel_mse_vs_output_rel_mse.png")
    write_report(summary_dir / "README.md", correlations, layer_summary, type_summary, method_summary)
    write_json(
        summary_dir / "metadata.json",
        {
            "local_rows": len(local_rows),
            "ablation_rows": len(ablation_rows),
            "layer_type_rows": len(layer_type),
            "correlations": len(correlations),
        },
    )
    print(f"wrote {summary_dir}")


def load_ablation_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "ablations").glob("loss_ablation*.csv")):
        rows.extend(read_csv(path))
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        seen[(row["method"], row["policy"])] = row
    return list(seen.values())


def join_layer_type(local_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    local_by_key = {
        (row["method"], int(f(row, "layer")), row["module_type"]): row
        for row in local_rows
    }
    out = []
    for row in ablation_rows:
        if row.get("method") == "dense_bf16":
            continue
        key = (row["method"], int(f(row, "layer")), row["linear_type"])
        local = local_by_key.get(key)
        if not local:
            continue
        item = dict(row)
        item["module_name"] = local["module_name"]
        item["module_family"] = local["module_family"]
        item["numel"] = local["numel"]
        for metric in LOCAL_METRICS:
            item[metric] = local.get(metric, "")
        out.append(item)
    return out


def compute_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        out.extend(correlation_rows(method, method_rows))
    out.extend(correlation_rows("all", rows))
    return out


def correlation_rows(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    y = [f(row, "loss_delta_vs_dense") for row in rows]
    for metric in LOCAL_METRICS:
        x = [f(row, metric) for row in rows]
        out.append(
            {
                "method": method,
                "metric": metric,
                "n": len(rows),
                "pearson": pearson(x, y),
                "spearman": spearman(x, y),
            }
        )
    return out


def summarize_group(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("method") == "dense_bf16":
            continue
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out = []
    for key, items in groups.items():
        losses = [f(row, "loss_delta_vs_dense") for row in items]
        item = {name: value for name, value in zip(keys, key)}
        item.update(
            {
                "rows": len(items),
                "loss_delta_mean": mean(losses),
                "loss_delta_min": min(losses),
                "loss_delta_max": max(losses),
            }
        )
        out.append(item)
    return sorted(out, key=lambda row: tuple(row.get(key, "") for key in keys))


def plot_layer_depth(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in sorted({row["method"] for row in rows}):
        items = sorted([row for row in rows if row["method"] == method], key=lambda row: int(f(row, "layer")))
        ax.plot([int(f(row, "layer")) for row in items], [f(row, "loss_delta_mean") for row in items], marker="o", label=method)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Loss delta vs dense")
    ax.set_title("Layer-wise compression loss impact")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_type_loss(rows: list[dict[str, Any]], path: Path) -> None:
    methods = sorted({row["method"] for row in rows})
    types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, method in enumerate(methods):
        vals = []
        by_type = {row["linear_type"]: row for row in rows if row["method"] == method}
        for typ in types:
            vals.append(f(by_type.get(typ, {}), "loss_delta_mean", math.nan))
        xs = [i + (idx - (len(methods) - 1) / 2) * width for i in range(len(types))]
        ax.bar(xs, vals, width=width, label=method)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=25, ha="right")
    ax.set_ylabel("Loss delta vs dense")
    ax.set_title("Linear-type compression loss impact")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_best_proxy(rows: list[dict[str, Any]], correlations: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        write_empty_plot(path, "No layer-type rows available for local proxy correlation")
        return
    all_corr = [row for row in correlations if row["method"] == "all"]
    best = max(all_corr, key=lambda row: abs(f(row, "spearman"))) if all_corr else {"metric": "output_rel_mse"}
    metric = best["metric"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in sorted({row["method"] for row in rows}):
        items = [row for row in rows if row["method"] == method]
        ax.scatter([f(row, metric) for row in items], [f(row, "loss_delta_vs_dense") for row in items], label=method, alpha=0.75)
    ax.set_xlabel(metric)
    ax.set_ylabel("Measured loss delta vs dense")
    ax.set_title(f"Best local proxy vs measured loss delta ({metric})")
    ax.grid(True, alpha=0.3)
    if rows:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_weight_vs_output(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in sorted({row["method"] for row in rows}):
        items = [row for row in rows if row["method"] == method]
        ax.scatter([f(row, "weight_rel_mse") for row in items], [f(row, "output_rel_mse") for row in items], label=method, alpha=0.7)
    ax.set_xlabel("weight_rel_mse")
    ax.set_ylabel("output_rel_mse")
    ax.set_title("Weight error vs activation-conditioned output error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    correlations: list[dict[str, Any]],
    layer_summary: list[dict[str, Any]],
    type_summary: list[dict[str, Any]],
    method_summary: list[dict[str, Any]],
) -> None:
    best = sorted([row for row in correlations if row["method"] == "all"], key=lambda row: abs(f(row, "spearman")), reverse=True)
    worst_layers = sorted(layer_summary, key=lambda row: f(row, "loss_delta_mean"), reverse=True)[:8]
    worst_types = sorted(type_summary, key=lambda row: f(row, "loss_delta_mean"), reverse=True)[:8]
    lines = [
        "# Llama2 Prefill-Only Loss Modeling Summary",
        "",
        "This report summarizes local linear errors and WikiText-2 mean CE loss deltas for prefill-only precision modeling.",
        "",
        "## Best Local Error Proxies",
        "",
        "| metric | Pearson | Spearman | n |",
        "|---|---:|---:|---:|",
    ]
    if best:
        for row in best[:6]:
            lines.append(f"| {row['metric']} | {f(row, 'pearson'):.4f} | {f(row, 'spearman'):.4f} | {int(f(row, 'n'))} |")
    else:
        lines.append("| n/a | n/a | n/a | 0 |")
    lines.extend(["", "## Highest Loss-Delta Layers", "", "| method | layer | mean loss delta |", "|---|---:|---:|"])
    for row in worst_layers:
        lines.append(f"| {row['method']} | {row['layer']} | {f(row, 'loss_delta_mean'):.6f} |")
    lines.extend(["", "## Highest Loss-Delta Linear Types", "", "| method | type | mean loss delta |", "|---|---|---:|"])
    for row in worst_types:
        lines.append(f"| {row['method']} | {row['linear_type']} | {f(row, 'loss_delta_mean'):.6f} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `local_error_loss_correlations.csv`",
            "- `layer_loss_summary.csv`",
            "- `linear_type_loss_summary.csv`",
            "- `layer_depth_loss_delta.png`",
            "- `linear_type_loss_delta.png`",
            "- `best_local_proxy_vs_loss_delta.png`",
            "- `weight_rel_mse_vs_output_rel_mse.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pearson(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return math.nan
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in pairs)
    den_x = sum((a - mx) ** 2 for a in xs) ** 0.5
    den_y = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / den_x / den_y if den_x > 0 and den_y > 0 else math.nan


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rank(x), rank(y))


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + j - 1) / 2 + 1
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


if __name__ == "__main__":
    main()
