#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from common_sparse_bf16_proxy import DEBUG_ROOT, LOCAL_ERROR_METRIC, SOURCE_014_ROOT, f, read_csv, selected_from_text, write_csv
from generate_controlled_proxy_pairs import SOURCE_015_ROOT, load_local_rows


METHODS = ("sparse_bf16", "dense_nvfp4", "sparse_nvfp4")
LINEAR_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LAYERS = tuple(range(32))
VARIANTS = ("raw_only", "layer_depth", "linear_type", "layer_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validated controlled structural ablation on raw-matched pair deltas.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-014-root", type=Path, default=SOURCE_014_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--tag", default="controlled")
    parser.add_argument("--ridge", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    pred_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        examples = build_pair_examples(args, method)
        for variant in VARIANTS:
            rows = loo_predictions(method, variant, examples, ridge=args.ridge)
            pred_rows.extend(rows)
            summary_rows.append(summarize(method, variant, rows))

    out = args.output_root / "controlled"
    write_csv(out / "controlled_structure_ablation_predictions.csv", pred_rows)
    write_csv(out / "controlled_structure_ablation_summary.csv", summary_rows)
    plot_predictions(pred_rows, out / "controlled_structure_ablation_pair_delta.png")
    write_report(out / "controlled_structure_ablation_summary.md", summary_rows)
    print(f"wrote {out / 'controlled_structure_ablation_summary.md'}")


def build_pair_examples(args: argparse.Namespace, method: str) -> list[dict[str, Any]]:
    local_args = argparse.Namespace(source_014_root=args.source_014_root, source_015_root=args.source_015_root, metric=args.metric)
    local_rows = load_local_rows(local_args, method)
    module_info = {
        row["module_name"]: (int(f(row, "layer")), row["module_type"], f(row, args.metric))
        for row in local_rows
    }
    policies = read_csv(args.output_root / "controlled" / "policies" / f"controlled_policies_{method}.csv")
    losses = read_csv(args.output_root / "loss" / f"loss_samples_{method}_{args.tag}.csv")
    loss_by_policy = {row["policy_id"]: f(row, "loss_delta_vs_dense") for row in losses}
    arms_by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in policies:
        arms_by_pair.setdefault(row["pair_id"], {})[row["arm"]] = row

    examples = []
    for pair_id, arms in sorted(arms_by_pair.items()):
        low = arms["low_final"]
        high = arms["high_final"]
        layer_delta = np.zeros(len(LAYERS), dtype=np.float64)
        type_delta = np.zeros(len(LINEAR_TYPES), dtype=np.float64)
        cell_delta = np.zeros((len(LAYERS), len(LINEAR_TYPES)), dtype=np.float64)
        raw_delta = 0.0
        for arm, sign in ((high, 1.0), (low, -1.0)):
            for name in selected_from_text(arm["selected_names"]):
                layer, typ, error = module_info[name]
                layer_idx = LAYERS.index(layer)
                type_idx = LINEAR_TYPES.index(typ)
                raw_delta += sign * error
                layer_delta[layer_idx] += sign * error
                type_delta[type_idx] += sign * error
                cell_delta[layer_idx, type_idx] += sign * error
        y = loss_by_policy[high["policy_id"]] - loss_by_policy[low["policy_id"]]
        examples.append(
            {
                "method": method,
                "pair_id": pair_id,
                "selected_modules": int(f(low, "selected_modules")),
                "raw_delta": raw_delta,
                "layer_delta": layer_delta,
                "type_delta": type_delta,
                "cell_delta": cell_delta.reshape(-1),
                "loss_delta": y,
            }
        )
    return examples


def loo_predictions(method: str, variant: str, examples: list[dict[str, Any]], *, ridge: float) -> list[dict[str, Any]]:
    x = np.stack([features(row, variant) for row in examples])
    y = np.array([row["loss_delta"] for row in examples], dtype=np.float64)
    out = []
    for holdout in range(len(examples)):
        train = np.array([idx for idx in range(len(examples)) if idx != holdout])
        coef = fit_ridge(x[train], y[train], ridge=ridge)
        pred = float(np.r_[1.0, x[holdout]] @ coef)
        row = examples[holdout]
        out.append(
            {
                "method": method,
                "variant": variant,
                "pair_id": row["pair_id"],
                "selected_modules": row["selected_modules"],
                "raw_pair_delta": row["raw_delta"],
                "loss_delta": row["loss_delta"],
                "pred_pair_delta": pred,
                "abs_error": abs(row["loss_delta"] - pred),
                "direction_correct": int((row["loss_delta"] > 0) == (pred > 0)),
            }
        )
    return out


def features(row: dict[str, Any], variant: str) -> np.ndarray:
    if variant == "raw_only":
        return np.array([row["raw_delta"]], dtype=np.float64)
    if variant == "layer_depth":
        return np.r_[row["raw_delta"], row["layer_delta"]]
    if variant == "linear_type":
        return np.r_[row["raw_delta"], row["type_delta"]]
    if variant == "layer_type":
        return np.r_[row["raw_delta"], row["layer_delta"], row["type_delta"]]
    raise ValueError(variant)


def fit_ridge(x: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray:
    x_aug = np.c_[np.ones(len(x)), x]
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)


def summarize(method: str, variant: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = [f(row, "loss_delta") for row in rows]
    pred = [f(row, "pred_pair_delta") for row in rows]
    return {
        "method": method,
        "variant": variant,
        "pairs": len(rows),
        "loo_pearson": pearson(pred, y),
        "loo_spearman": spearman(pred, y),
        "loo_mae": mean(f(row, "abs_error") for row in rows),
        "direction_accuracy": mean(f(row, "direction_correct") for row in rows),
    }


def plot_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    methods = sorted({row["method"] for row in rows})
    variants = list(VARIANTS)
    fig, axes = plt.subplots(len(methods), len(variants), figsize=(4.2 * len(variants), 3.5 * len(methods)), squeeze=False)
    for i, method in enumerate(methods):
        for j, variant in enumerate(variants):
            ax = axes[i][j]
            items = [row for row in rows if row["method"] == method and row["variant"] == variant]
            xs = [f(row, "pred_pair_delta") for row in items]
            ys = [f(row, "loss_delta") for row in items]
            ax.scatter(xs, ys, alpha=0.85)
            if xs and ys:
                lo = min(min(xs), min(ys))
                hi = max(max(xs), max(ys))
                ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.45)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.25)
            ax.axvline(0, color="black", linewidth=0.8, alpha=0.25)
            ax.set_title(f"{method} / {variant}")
            ax.set_xlabel("LOO predicted pair delta")
            ax.set_ylabel("Measured pair delta")
            ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Controlled Structure Ablation",
        "",
        "Leave-one-pair-out results on raw-local-matched controlled pairs. The target is `loss(high_final) - loss(low_final)`.",
        "",
        "| method | variant | pairs | Pearson | Spearman | MAE | direction acc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['variant']} | {int(f(row, 'pairs'))} | "
            f"{f(row, 'loo_pearson'):.4f} | {f(row, 'loo_spearman'):.4f} | "
            f"{f(row, 'loo_mae'):.6f} | {f(row, 'direction_accuracy'):.4f} |"
        )
    lines.extend(["", "## Plot", "", f"- `{path.parent / 'controlled_structure_ablation_pair_delta.png'}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / den_x / den_y if den_x and den_y else math.nan


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
