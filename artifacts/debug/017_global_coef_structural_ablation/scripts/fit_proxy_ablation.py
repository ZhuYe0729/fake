#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import torch

from common_sparse_bf16_proxy import (
    DEBUG_ROOT,
    FAKE_ROOT,
    LAYERS,
    LINEAR_TYPES,
    LOCAL_ERROR_METRIC,
    SOURCE_014_ROOT,
    f,
    load_sparse_local_errors,
    policy_paths,
    read_csv,
    selected_from_text,
    write_csv,
    write_json,
)


SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"
METHODS = ("sparse_bf16", "dense_nvfp4", "sparse_nvfp4")
VARIANTS = ("local_only", "local_layer", "local_type", "final_layer_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural ablations for precision proxy models.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-014-root", type=Path, default=SOURCE_014_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--calibration-l2", type=float, default=1e-6)
    parser.add_argument("--policies-csv-template", default="")
    parser.add_argument("--loss-tag", default="")
    parser.add_argument("--output-subdir", default="ablation")
    parser.add_argument("--expected-examples", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    all_predictions: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    coefficients: dict[str, Any] = {}

    for method in methods:
        policies_csv = policy_csv_for_method(args, method)
        policies = {row["policy_id"]: row for row in read_csv(policies_csv)}
        examples = load_examples(args, method, policies)
        assign_splits(examples, train_ratio=args.train_ratio, seed=args.seed)
        coefficients[method] = {}
        for variant in VARIANTS:
            fit = fit_variant(examples, method=method, variant=variant, metric=args.metric, steps=args.steps, lr=args.lr, l2=args.l2)
            predictions = add_predictions(examples, fit)
            metrics = metric_rows(predictions)
            all_predictions.extend(predictions)
            all_metrics.extend(metrics)
            coefficients[method][variant] = fit
        if method == "dense_nvfp4":
            ref_predictions, ref_fit = dense_calibrated_reference(examples, all_predictions, l2=args.calibration_l2)
            ref_metrics = metric_rows(ref_predictions)
            all_predictions.extend(ref_predictions)
            all_metrics.extend(ref_metrics)
            coefficients[method]["dense_calibrated_reference"] = ref_fit

    paths = ablation_paths(args.output_root, args.output_subdir)
    write_csv(paths["predictions"], all_predictions)
    write_csv(paths["metrics"], all_metrics)
    write_json(paths["coefficients"], coefficients)
    plot_metric(all_metrics, metric="spearman", split="holdout", path=paths["spearman_plot"])
    plot_metric(all_metrics, metric="rmse", split="holdout", path=paths["rmse_plot"])
    for method in methods:
        plot_holdout_predictions(
            all_predictions,
            all_metrics,
            method=method,
            path=paths["prediction_plot_dir"] / f"proxy_ablation_holdout_predictions_{method}.png",
        )
    write_summary(paths["summary"], all_metrics, paths)
    print(f"wrote {paths['metrics']}")
    print(f"wrote {paths['summary']}")


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={METHODS}")
    return methods


def policy_csv_for_method(args: argparse.Namespace, method: str) -> Path:
    if args.policies_csv_template:
        return Path(args.policies_csv_template.format(method=method))
    return policy_paths(args.output_root)["policies"]


def ablation_paths(output_root: Path, output_subdir: str) -> dict[str, Path]:
    out = output_root / output_subdir
    return {
        "predictions": out / "proxy_ablation_predictions.csv",
        "metrics": out / "proxy_ablation_metrics.csv",
        "coefficients": out / "proxy_ablation_coefficients.json",
        "summary": out / "proxy_ablation_summary.md",
        "spearman_plot": out / "proxy_ablation_holdout_spearman.png",
        "rmse_plot": out / "proxy_ablation_holdout_rmse.png",
        "prediction_plot_dir": out,
    }


def load_examples(args: argparse.Namespace, method: str, policies: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    suffix = f"_{args.loss_tag}" if args.loss_tag else ""
    loss_rows = read_csv(args.output_root / "loss" / f"loss_samples_{method}{suffix}.csv")
    if method == "sparse_bf16":
        local_by_name = {row["module_name"]: row for row in load_sparse_local_errors(args.source_014_root)}
        local_source = str(args.source_014_root / "sensitivity" / "module_method_local_errors.csv")
    else:
        local_path = args.source_015_root / "sensitivity" / "module_method_kernel_local_errors.csv"
        local_by_name = {row["module_name"]: row for row in read_csv(local_path) if row.get("method") == method}
        local_source = str(local_path)
    if not local_by_name:
        raise RuntimeError(f"no local error rows for {method}")

    out = []
    for row in loss_rows:
        policy = policies.get(row["policy_id"])
        if policy is None:
            continue
        selected = selected_from_text(policy["selected_names"])
        terms = []
        for name in selected:
            local = local_by_name.get(name)
            if local is None:
                raise KeyError(f"missing local error for {method} {name}")
            terms.append((int(f(local, "layer")), local["module_type"], f(local, args.metric)))
        out.append(
            {
                "method": method,
                "policy_id": row["policy_id"],
                "sample_kind": row.get("sample_kind", policy.get("sample_kind", "")),
                "selected_modules": int(f(row, "selected_modules")),
                "loss_delta_vs_dense": f(row, "loss_delta_vs_dense"),
                "terms": terms,
                "local_error_source": local_source,
            }
        )
    if args.expected_examples > 0 and len(out) != args.expected_examples:
        raise RuntimeError(f"expected {args.expected_examples} examples for {method}, found {len(out)}")
    return out


def assign_splits(examples: list[dict[str, Any]], *, train_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    train_count = max(1, min(len(indices) - 1, round(len(indices) * train_ratio)))
    train = set(indices[:train_count])
    for idx, row in enumerate(examples):
        row["split"] = "train" if idx in train else "holdout"


def fit_variant(
    examples: list[dict[str, Any]],
    *,
    method: str,
    variant: str,
    metric: str,
    steps: int,
    lr: float,
    l2: float,
) -> dict[str, Any]:
    layer_index = {layer: idx for idx, layer in enumerate(LAYERS)}
    type_index = {typ: idx for idx, typ in enumerate(LINEAR_TYPES)}
    train_examples = [row for row in examples if row["split"] == "train"]
    x = design_tensor(train_examples, layer_index, type_index)
    y = torch.tensor([row["loss_delta_vs_dense"] for row in train_examples], dtype=torch.float64)
    bias = torch.tensor(float(y.mean().item()), dtype=torch.float64, requires_grad=True)
    params: list[torch.Tensor] = [bias]
    state: dict[str, torch.Tensor] = {"log_global": torch.zeros(1, dtype=torch.float64, requires_grad=True)}
    params.append(state["log_global"])
    if variant == "local_only":
        pass
    elif variant == "local_layer":
        state["log_layer"] = torch.zeros(len(LAYERS), dtype=torch.float64, requires_grad=True)
        params.append(state["log_layer"])
    elif variant == "local_type":
        state["log_type"] = torch.zeros(len(LINEAR_TYPES), dtype=torch.float64, requires_grad=True)
        params.append(state["log_type"])
    elif variant == "final_layer_type":
        state["log_layer"] = torch.zeros(len(LAYERS), dtype=torch.float64, requires_grad=True)
        state["log_type"] = torch.zeros(len(LINEAR_TYPES), dtype=torch.float64, requires_grad=True)
        params.extend([state["log_layer"], state["log_type"]])
    else:
        raise ValueError(variant)

    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = predict_tensor(x, variant, bias, state)
        penalty = sum(torch.mean(value**2) for value in state.values()) if state else torch.tensor(0.0, dtype=torch.float64)
        loss = torch.mean((pred - y) ** 2) + l2 * penalty
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_pred = predict_tensor(x, variant, bias, state)
        train_mse = float(torch.mean((train_pred - y) ** 2).item())
    return serialize_fit(method, variant, metric, bias, state, train_mse, examples[0]["local_error_source"])


def design_tensor(examples: list[dict[str, Any]], layer_index: dict[int, int], type_index: dict[str, int]) -> torch.Tensor:
    x = torch.zeros((len(examples), len(LAYERS), len(LINEAR_TYPES)), dtype=torch.float64)
    for row_idx, row in enumerate(examples):
        for layer, typ, error in row["terms"]:
            x[row_idx, layer_index[layer], type_index[typ]] += float(error)
    return x


def predict_tensor(x: torch.Tensor, variant: str, bias: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    global_coef = torch.exp(state["log_global"])[0]
    if variant == "local_only":
        return bias + global_coef * torch.sum(x, dim=(1, 2))
    if variant == "local_layer":
        log_layer = state["log_layer"] - state["log_layer"].mean()
        weights = global_coef * torch.exp(log_layer)[:, None]
        return bias + torch.sum(x * weights[None, :, :], dim=(1, 2))
    if variant == "local_type":
        log_type = state["log_type"] - state["log_type"].mean()
        weights = global_coef * torch.exp(log_type)[None, :]
        return bias + torch.sum(x * weights[None, :, :], dim=(1, 2))
    if variant == "final_layer_type":
        log_layer = state["log_layer"] - state["log_layer"].mean()
        log_type = state["log_type"] - state["log_type"].mean()
        weights = global_coef * torch.exp(log_layer)[:, None] * torch.exp(log_type)[None, :]
        return bias + torch.sum(x * weights[None, :, :], dim=(1, 2))
    raise ValueError(variant)


def serialize_fit(
    method: str,
    variant: str,
    metric: str,
    bias: torch.Tensor,
    state: dict[str, torch.Tensor],
    train_mse: float,
    local_source: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": method,
        "variant": variant,
        "local_error_metric": metric,
        "local_error_source": local_source,
        "bias": float(bias.detach().item()),
        "train_mse": train_mse,
    }
    if "log_global" in state:
        out["global_coef"] = float(torch.exp(state["log_global"].detach())[0].item())
    if "log_layer" in state:
        log_layer = state["log_layer"].detach() - state["log_layer"].detach().mean()
        layer_coef = torch.exp(log_layer)
        out["layer_coef"] = {str(layer): float(layer_coef[idx].item()) for idx, layer in enumerate(LAYERS)}
    if "log_type" in state:
        log_type = state["log_type"].detach() - state["log_type"].detach().mean()
        type_coef = torch.exp(log_type)
        out["type_coef"] = {typ: float(type_coef[idx].item()) for idx, typ in enumerate(LINEAR_TYPES)}
    return out


def add_predictions(examples: list[dict[str, Any]], fit: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in examples:
        pred = fit["bias"]
        raw_error_sum = 0.0
        for layer, typ, error in row["terms"]:
            raw_error_sum += error
            pred += error * coefficient_for(fit, layer, typ)
        out.append(
            {
                "method": row["method"],
                "variant": fit["variant"],
                "policy_id": row["policy_id"],
                "split": row["split"],
                "sample_kind": row["sample_kind"],
                "selected_modules": row["selected_modules"],
                "loss_delta_vs_dense": row["loss_delta_vs_dense"],
                "pred_loss_delta": pred,
                "raw_error_sum": raw_error_sum,
                "residual": row["loss_delta_vs_dense"] - pred,
            }
        )
    return sorted(out, key=lambda item: (item["method"], item["variant"], item["split"], item["policy_id"]))


def coefficient_for(fit: dict[str, Any], layer: int, typ: str) -> float:
    variant = fit["variant"]
    if variant == "local_only":
        return fit["global_coef"]
    if variant == "local_layer":
        return fit["global_coef"] * fit["layer_coef"][str(layer)]
    if variant == "local_type":
        return fit["global_coef"] * fit["type_coef"][typ]
    if variant == "final_layer_type":
        return fit["global_coef"] * fit["layer_coef"][str(layer)] * fit["type_coef"][typ]
    raise ValueError(variant)


def dense_calibrated_reference(
    examples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    l2: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = [row for row in predictions if row["method"] == "dense_nvfp4" and row["variant"] == "final_layer_type"]
    train_rows = [row for row in base_rows if row["split"] == "train"]
    x = torch.tensor([dense_calibration_features(row) for row in train_rows], dtype=torch.float64)
    y = torch.tensor([f(row, "loss_delta_vs_dense") for row in train_rows], dtype=torch.float64)
    penalty = torch.eye(x.shape[1], dtype=torch.float64) * l2
    penalty[0, 0] = 0.0
    coef = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
    coeffs = {
        "intercept": float(coef[0].item()),
        "base_pred": float(coef[1].item()),
        "base_pred_squared": float(coef[2].item()),
        "log1p_selected_modules": float(coef[3].item()),
    }
    ref = []
    for row in base_rows:
        features = dense_calibration_features(row)
        weights = [coeffs["intercept"], coeffs["base_pred"], coeffs["base_pred_squared"], coeffs["log1p_selected_modules"]]
        pred = sum(weight * value for weight, value in zip(weights, features))
        item = dict(row)
        item["variant"] = "dense_calibrated_reference"
        item["base_pred_loss_delta"] = row["pred_loss_delta"]
        item["pred_loss_delta"] = pred
        item["residual"] = f(row, "loss_delta_vs_dense") - pred
        ref.append(item)
    fit = {
        "method": "dense_nvfp4",
        "variant": "dense_calibrated_reference",
        "formula": "c0 + c1 * final_layer_type_pred + c2 * final_layer_type_pred^2 + c3 * log1p(selected_modules)",
        "l2": l2,
        "coefficients": coeffs,
    }
    return ref, fit


def dense_calibration_features(row: dict[str, Any]) -> list[float]:
    base = f(row, "pred_loss_delta")
    return [1.0, base, base * base, math.log1p(max(f(row, "selected_modules"), 0.0))]


def metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    if not rows:
        return out
    method = rows[0]["method"]
    variant = rows[0]["variant"]
    for split in ("train", "holdout", "all"):
        items = rows if split == "all" else [row for row in rows if row["split"] == split]
        if not items:
            continue
        y = [f(row, "loss_delta_vs_dense") for row in items]
        p = [f(row, "pred_loss_delta") for row in items]
        residual = [a - b for a, b in zip(y, p)]
        out.append(
            {
                "method": method,
                "variant": variant,
                "split": split,
                "rows": len(items),
                "pearson": pearson(p, y),
                "spearman": spearman(p, y),
                "mae": mean(abs(value) for value in residual),
                "rmse": math.sqrt(mean(value * value for value in residual)),
            }
        )
    return out


def plot_metric(rows: list[dict[str, Any]], *, metric: str, split: str, path: Path) -> None:
    variants = list(VARIANTS) + ["dense_calibrated_reference"]
    methods = list(METHODS)
    by_key = {(row["method"], row["variant"], row["split"]): f(row, metric, math.nan) for row in rows}
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 5))
    for method_idx, method in enumerate(methods):
        xs = [i + (method_idx - 1) * width for i in range(len(variants))]
        vals = [by_key.get((method, variant, split), math.nan) for variant in variants]
        ax.bar(xs, vals, width=width, label=method)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel(f"{split} {metric}")
    ax.set_title(f"Proxy ablation {split} {metric}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_holdout_predictions(
    rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    *,
    method: str,
    path: Path,
) -> None:
    variants = list(VARIANTS)
    if method == "dense_nvfp4":
        variants.append("dense_calibrated_reference")
    metric_by_variant = {
        row["variant"]: row
        for row in metrics
        if row["method"] == method and row["split"] == "holdout"
    }
    ncols = 3 if len(variants) > 4 else 2
    nrows = math.ceil(len(variants) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows), squeeze=False, constrained_layout=True)
    all_holdout = [row for row in rows if row["method"] == method and row["split"] == "holdout" and row["variant"] in variants]
    counts = [f(row, "selected_modules") for row in all_holdout]
    vmin, vmax = (min(counts), max(counts)) if counts else (0.0, 1.0)
    scatter_for_colorbar = None
    for idx, variant in enumerate(variants):
        ax = axes[idx // ncols][idx % ncols]
        items = [row for row in rows if row["method"] == method and row["variant"] == variant and row["split"] == "holdout"]
        xs = [f(row, "pred_loss_delta") for row in items]
        ys = [f(row, "loss_delta_vs_dense") for row in items]
        cs = [f(row, "selected_modules") for row in items]
        scatter_for_colorbar = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.85, edgecolors="none")
        if items:
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            pad = (hi - lo) * 0.05 if hi > lo else 0.01
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1, alpha=0.45)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        metric = metric_by_variant.get(variant, {})
        ax.set_title(
            f"{variant}\n"
            f"rho={f(metric, 'spearman', math.nan):.3f}, RMSE={f(metric, 'rmse', math.nan):.4f}"
        )
        ax.set_xlabel("Predicted loss delta")
        ax.set_ylabel("Measured loss delta")
        ax.grid(True, alpha=0.3)
    for idx in range(len(variants), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_axis_off()
    if scatter_for_colorbar is not None:
        fig.colorbar(scatter_for_colorbar, ax=axes.ravel().tolist(), shrink=0.85, label="selected_modules")
    fig.suptitle(f"{method} holdout predicted vs measured loss")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    holdout = [row for row in rows if row["split"] == "holdout"]
    lines = [
        "# Precision Proxy Ablation Summary",
        "",
        "## Holdout Metrics",
        "",
        "| method | variant | rows | Pearson | Spearman | MAE | RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(holdout, key=lambda item: (item["method"], variant_order(item["variant"]))):
        lines.append(
            f"| {row['method']} | {row['variant']} | {int(f(row, 'rows'))} | {f(row, 'pearson'):.4f} | "
            f"{f(row, 'spearman'):.4f} | {f(row, 'mae'):.6f} | {f(row, 'rmse'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"- `{paths['spearman_plot']}`",
            f"- `{paths['rmse_plot']}`",
            f"- `{paths['prediction_plot_dir'] / 'proxy_ablation_holdout_predictions_sparse_bf16.png'}`",
            f"- `{paths['prediction_plot_dir'] / 'proxy_ablation_holdout_predictions_dense_nvfp4.png'}`",
            f"- `{paths['prediction_plot_dir'] / 'proxy_ablation_holdout_predictions_sparse_nvfp4.png'}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def variant_order(value: str) -> int:
    order = {name: idx for idx, name in enumerate(list(VARIANTS) + ["dense_calibrated_reference"])}
    return order.get(value, 999)


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    px, py = zip(*pairs)
    mx, my = mean(px), mean(py)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in px))
    den_y = math.sqrt(sum((y - my) ** 2 for y in py))
    return num / den_x / den_y if den_x and den_y else math.nan


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


if __name__ == "__main__":
    main()
