#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from ablate_sparse_nvfp4_empirical_scenarios import (
    SOURCE_015_ROOT,
    VARIANTS,
    feature_dict,
    fit_variant,
    load_modules,
    load_stratified_examples,
    predict_features,
)
from common_sparse_bf16_proxy import DEBUG_ROOT, LOCAL_ERROR_METRIC, f, read_csv, selected_from_text, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot measured test loss vs ablation predicted loss for sparse NVFP4 scenarios.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--scenario-prefix", default="sparse_nvfp4_empirical_balanced_scenario")
    parser.add_argument("--loss-tag", default="empirical_balanced")
    parser.add_argument("--output-prefix", default="sparse_nvfp4_empirical_balanced_config_loss_ablation")
    parser.add_argument("--ridge", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = load_modules(args)
    module_by_name = {row["name"]: row for row in modules}
    train_examples = load_stratified_examples(args, module_by_name)
    test_configs = load_test_configs(args, module_by_name)

    pred_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        fit = fit_variant(train_examples, variant, ridge=args.ridge)
        rows = []
        for config in test_configs:
            pred = predict_features(config["features"], fit)
            item = {
                "variant": variant,
                "policy_id": config["policy_id"],
                "pair_id": config["pair_id"],
                "side": config["side"],
                "measured_loss_delta": config["measured_loss_delta"],
                "measured_loss": config["measured_loss"],
                "predicted_loss_delta": pred,
                "predicted_loss": config["dense_loss"] + pred,
                "dense_loss": config["dense_loss"],
            }
            item["abs_error_delta"] = abs(item["predicted_loss_delta"] - item["measured_loss_delta"])
            rows.append(item)
        pred_rows.extend(rows)
        summary_rows.append(summarize(variant, rows))

    out = args.output_root / "structural_scenarios"
    write_csv(out / f"{args.output_prefix}_predictions.csv", pred_rows)
    write_csv(out / f"{args.output_prefix}_summary.csv", summary_rows)
    plot_loss(pred_rows, out / f"{args.output_prefix}_scatter.png")
    plot_metrics(summary_rows, out / f"{args.output_prefix}_metrics.png")
    write_report(out / f"{args.output_prefix}_summary.md", summary_rows, out, args.output_prefix)
    print(f"wrote {out / f'{args.output_prefix}_summary.md'}")


def load_test_configs(args: argparse.Namespace, module_by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    policies = {row["policy_id"]: row for row in read_csv(args.output_root / "structural_scenarios" / f"{args.scenario_prefix}_policies.csv")}
    losses = read_csv(args.output_root / "loss" / f"loss_samples_sparse_nvfp4_{args.loss_tag}.csv")
    out = []
    for row in losses:
        policy_id = row["policy_id"]
        pair_id, side = split_policy_id(policy_id)
        out.append(
            {
                "policy_id": policy_id,
                "pair_id": pair_id,
                "side": side,
                "measured_loss_delta": f(row, "loss_delta_vs_dense"),
                "measured_loss": f(row, "loss"),
                "dense_loss": f(row, "dense_loss"),
                "features": feature_dict(selected_from_text(policies[policy_id]["selected_names"]), module_by_name),
            }
        )
    return sorted(out, key=lambda row: (row["pair_id"], row["side"]))


def split_policy_id(policy_id: str) -> tuple[str, str]:
    if policy_id.endswith("_low_empirical"):
        return policy_id.removesuffix("_low_empirical"), "low"
    if policy_id.endswith("_high_empirical"):
        return policy_id.removesuffix("_high_empirical"), "high"
    return policy_id, "unknown"


def summarize(variant: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred = [f(row, "predicted_loss_delta") for row in rows]
    measured = [f(row, "measured_loss_delta") for row in rows]
    errors = [f(row, "abs_error_delta") for row in rows]
    return {
        "variant": variant,
        "configs": len(rows),
        "pearson": pearson(pred, measured),
        "spearman": spearman(pred, measured),
        "mae": mean(errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
        "predicted_delta_mean": mean(pred),
        "measured_delta_mean": mean(measured),
    }


def plot_loss(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.1 * len(VARIANTS), 3.8), squeeze=False)
    for idx, variant in enumerate(VARIANTS):
        ax = axes[0][idx]
        items = [row for row in rows if row["variant"] == variant]
        xs = [f(row, "predicted_loss_delta") for row in items]
        ys = [f(row, "measured_loss_delta") for row in items]
        colors = ["#3b82f6" if row["side"] == "low" else "#ef4444" for row in items]
        ax.scatter(xs, ys, c=colors, alpha=0.85, s=32)
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = (hi - lo) * 0.08 if hi > lo else 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1, alpha=0.45)
        ax.set_title(variant)
        ax.set_xlabel("Predicted loss delta")
        ax.set_ylabel("Measured test loss delta")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    pearson_values = [float(row["pearson"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, mae)
    axes[0].set_title("Config loss MAE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, pearson_values)
    axes[1].set_title("Config loss Pearson")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], out_dir: Path, output_prefix: str) -> None:
    lines = [
        "# Sparse NVFP4 Balanced Config-Level Loss Prediction Ablation",
        "",
        "Each ablation variant is fitted on the stratified sparse NVFP4 samples and evaluated on the balanced structural scenario configs.",
        "",
        "| variant | configs | Pearson | Spearman | MAE | RMSE | pred delta mean | measured delta mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {int(float(row['configs']))} | {float(row['pearson']):.4f} | "
            f"{float(row['spearman']):.4f} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['predicted_delta_mean']):.6f} | {float(row['measured_delta_mean']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"- `{out_dir / f'{output_prefix}_scatter.png'}`",
            f"- `{out_dir / f'{output_prefix}_metrics.png'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float:
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
