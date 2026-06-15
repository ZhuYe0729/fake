#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from common_sparse_bf16_proxy import DEBUG_ROOT, FAKE_ROOT, LOCAL_ERROR_METRIC, f, read_csv, selected_from_text, write_csv


SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"
METHOD = "sparse_nvfp4"
TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BUCKETS = ((0, 7, "L0_7"), (8, 15, "L8_15"), (16, 23, "L16_23"), (24, 31, "L24_31"))
VARIANTS = ("local_only", "local_depth", "local_type", "final_depth_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate sparse NVFP4 empirical structural scenarios.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--scenario-prefix", default="sparse_nvfp4_empirical_scenario")
    parser.add_argument("--loss-tag", default="empirical_structural")
    parser.add_argument("--output-prefix", default="sparse_nvfp4_empirical_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = load_modules(args)
    module_by_name = {row["name"]: row for row in modules}
    train_examples = load_stratified_examples(args, module_by_name)
    scenario_pairs = load_scenario_pairs(args, module_by_name)

    predictions: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        coef = fit_variant(train_examples, variant, ridge=args.ridge)
        rows = predict_pairs(scenario_pairs, variant, coef)
        predictions.extend(rows)
        summary.append(summarize(variant, rows))

    out = args.output_root / "structural_scenarios"
    write_csv(out / f"{args.output_prefix}_predictions.csv", predictions)
    write_csv(out / f"{args.output_prefix}_summary.csv", summary)
    plot_ablation(predictions, out / f"{args.output_prefix}_loss_delta.png")
    plot_bars(summary, out / f"{args.output_prefix}_metrics.png")
    write_report(out / f"{args.output_prefix}_summary.md", summary, out, args.output_prefix)
    print(f"wrote {out / f'{args.output_prefix}_summary.md'}")


def load_modules(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.source_015_root / "sensitivity" / "module_method_kernel_local_errors.csv"
    rows = [row for row in read_csv(path) if row.get("method") == METHOD]
    return [
        {"name": row["module_name"], "layer": int(f(row, "layer")), "type": row["module_type"], "error": f(row, args.metric)}
        for row in rows
    ]


def load_stratified_examples(args: argparse.Namespace, module_by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    policies = {row["policy_id"]: row for row in read_csv(args.output_root / "stratified" / "policies" / "stratified_policies_sparse_nvfp4.csv")}
    losses = read_csv(args.output_root / "loss" / "loss_samples_sparse_nvfp4_stratified.csv")
    out = []
    for row in losses:
        policy = policies[row["policy_id"]]
        out.append(
            {
                "policy_id": row["policy_id"],
                "loss_delta": f(row, "loss_delta_vs_dense"),
                "features": feature_dict(selected_from_text(policy["selected_names"]), module_by_name),
            }
        )
    return out


def load_scenario_pairs(args: argparse.Namespace, module_by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = read_csv(args.output_root / "structural_scenarios" / f"{args.scenario_prefix}_pairs.csv")
    policies = {row["policy_id"]: row for row in read_csv(args.output_root / "structural_scenarios" / f"{args.scenario_prefix}_policies.csv")}
    losses = {row["policy_id"]: row for row in read_csv(args.output_root / "loss" / f"loss_samples_sparse_nvfp4_{args.loss_tag}.csv")}
    out = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        low_id = f"{pair_id}_low_empirical"
        high_id = f"{pair_id}_high_empirical"
        low_features = feature_dict(selected_from_text(policies[low_id]["selected_names"]), module_by_name)
        high_features = feature_dict(selected_from_text(policies[high_id]["selected_names"]), module_by_name)
        out.append(
            {
                "pair_id": pair_id,
                "raw_rel_gap": f(pair, "raw_rel_gap"),
                "low_features": low_features,
                "high_features": high_features,
                "measured_delta": f(losses[high_id], "loss_delta_vs_dense") - f(losses[low_id], "loss_delta_vs_dense"),
            }
        )
    return out


def feature_dict(names: set[str], module_by_name: dict[str, dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {"count": float(len(names)), "raw": 0.0}
    features.update({typ: 0.0 for typ in TYPES})
    features.update({bucket: 0.0 for _, _, bucket in BUCKETS})
    for name in names:
        module = module_by_name[name]
        error = module["error"]
        features["raw"] += error
        features[module["type"]] += error
        for lo, hi, bucket in BUCKETS:
            if lo <= module["layer"] <= hi:
                features[bucket] += error
    return features


def labels_for_variant(variant: str) -> list[str]:
    if variant == "local_only":
        return ["count", "raw"]
    if variant == "local_depth":
        return ["count", "raw", *(bucket for _, _, bucket in BUCKETS)]
    if variant == "local_type":
        return ["count", "raw", *TYPES]
    if variant == "final_depth_type":
        return ["count", "raw", *TYPES, *(bucket for _, _, bucket in BUCKETS)]
    raise ValueError(variant)


def fit_variant(examples: list[dict[str, Any]], variant: str, *, ridge: float) -> dict[str, Any]:
    labels = labels_for_variant(variant)
    x = np.array([[row["features"][label] for label in labels] for row in examples], dtype=np.float64)
    y = np.array([row["loss_delta"] for row in examples], dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    z = np.c_[np.ones(len(x)), (x - mu) / sd]
    penalty = np.eye(z.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return {"labels": labels, "mean": mu, "std": sd, "coef": coef}


def predict_features(features: dict[str, float], fit: dict[str, Any]) -> float:
    x = np.array([features[label] for label in fit["labels"]], dtype=np.float64)
    z = np.r_[1.0, (x - fit["mean"]) / fit["std"]]
    return float(z @ fit["coef"])


def predict_pairs(pairs: list[dict[str, Any]], variant: str, fit: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in pairs:
        low_pred = predict_features(row["low_features"], fit)
        high_pred = predict_features(row["high_features"], fit)
        pred_delta = high_pred - low_pred
        measured = row["measured_delta"]
        out.append(
            {
                "variant": variant,
                "pair_id": row["pair_id"],
                "raw_rel_gap": row["raw_rel_gap"],
                "pred_delta": pred_delta,
                "measured_delta": measured,
                "abs_error": abs(pred_delta - measured),
                "direction_correct": int((pred_delta > 0) == (measured > 0)),
            }
        )
    return out


def summarize(variant: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred = [f(row, "pred_delta") for row in rows]
    measured = [f(row, "measured_delta") for row in rows]
    return {
        "variant": variant,
        "pairs": len(rows),
        "pearson": pearson(pred, measured),
        "spearman": spearman(pred, measured),
        "mae": mean(f(row, "abs_error") for row in rows),
        "rmse": math.sqrt(mean(f(row, "abs_error") ** 2 for row in rows)),
        "direction_accuracy": mean(f(row, "direction_correct") for row in rows),
        "pred_delta_mean": mean(pred),
        "measured_delta_mean": mean(measured),
    }


def plot_ablation(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.2 * len(VARIANTS), 3.8), squeeze=False)
    for idx, variant in enumerate(VARIANTS):
        ax = axes[0][idx]
        items = [row for row in rows if row["variant"] == variant]
        xs = [f(row, "pred_delta") for row in items]
        ys = [f(row, "measured_delta") for row in items]
        ax.scatter(xs, ys, alpha=0.85)
        lo = min(min(xs), min(ys), 0.0)
        hi = max(max(xs), max(ys), 0.0)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.45)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.25)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.25)
        ax.set_title(variant)
        ax.set_xlabel("Predicted pair loss delta")
        ax.set_ylabel("Measured pair loss delta")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bars(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["variant"] for row in rows]
    spearman = [f(row, "spearman") for row in rows]
    mae = [f(row, "mae") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, spearman)
    axes[0].set_title("Spearman higher is better")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, mae)
    axes[1].set_title("MAE lower is better")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], out_dir: Path, output_prefix: str) -> None:
    lines = [
        "# Sparse NVFP4 Empirical Scenario Ablation",
        "",
        "Evaluation set: raw-local-matched sparse NVFP4 empirical structural pairs.",
        "",
        "| variant | pairs | Pearson | Spearman | MAE | RMSE | direction acc | pred delta mean | measured delta mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {int(f(row, 'pairs'))} | {f(row, 'pearson'):.4f} | {f(row, 'spearman'):.4f} | "
            f"{f(row, 'mae'):.6f} | {f(row, 'rmse'):.6f} | {f(row, 'direction_accuracy'):.4f} | "
            f"{f(row, 'pred_delta_mean'):.6f} | {f(row, 'measured_delta_mean'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"- `{out_dir / f'{output_prefix}_loss_delta.png'}`",
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
