#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
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
    f,
    policy_paths,
    read_csv,
    selected_from_text,
    write_csv,
    write_json,
)


SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"
METHODS = ("dense_nvfp4", "sparse_nvfp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit kernel-aware dense/sparse NVFP4 local-error proxy.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--dense-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration-l2", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    summary_rows = []
    for method in methods:
        metrics = fit_one_method(args, method)
        for row in metrics:
            item = dict(row)
            item["method"] = method
            summary_rows.append(item)
    write_combined_summary(args.output_root, summary_rows)


def fit_one_method(args: argparse.Namespace, method: str) -> list[dict[str, Any]]:
    paths = method_paths(args.output_root, method)
    policies = {row["policy_id"]: row for row in read_csv(policy_paths(args.output_root)["policies"])}
    loss_rows = read_csv(paths["loss"])
    local_by_name = {
        row["module_name"]: row
        for row in read_csv(args.source_015_root / "sensitivity" / "module_method_kernel_local_errors.csv")
        if row.get("method") == method
    }
    if not local_by_name:
        raise RuntimeError(f"No kernel local error rows for {method}")
    examples = build_examples(loss_rows, policies, local_by_name, args.metric)
    assign_splits(examples, train_ratio=args.train_ratio, seed=args.seed)
    fit = fit_model(examples, method=method, metric=args.metric, steps=args.steps, lr=args.lr, l2=args.l2)
    predictions = add_predictions(examples, fit)
    if method == "dense_nvfp4" and args.dense_calibration:
        calibration = fit_dense_calibration(predictions, l2=args.calibration_l2)
        apply_dense_calibration(predictions, calibration)
        fit["dense_calibration"] = calibration
    metrics = metric_rows(predictions)

    write_csv(paths["predictions"], predictions)
    write_csv(paths["metrics"], metrics)
    write_json(paths["model"], fit)
    plot_holdout(predictions, metrics, paths["plot"], method)
    write_method_summary(paths["summary"], method, fit, metrics, paths["plot"])
    print(f"wrote {method} model to {paths['model']}")
    print(f"wrote {method} holdout plot to {paths['plot']}")
    return metrics


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={METHODS}")
    return methods


def method_paths(output_root: Path, method: str) -> dict[str, Path]:
    return {
        "loss": output_root / "loss" / f"loss_samples_{method}.csv",
        "model": output_root / "model" / f"fitted_{method}_proxy.json",
        "predictions": output_root / "model" / f"predictions_{method}.csv",
        "metrics": output_root / "model" / f"proxy_metrics_{method}.csv",
        "plot": output_root / "plots" / f"holdout_{method}_proxy_vs_loss_delta.png",
        "summary": output_root / "summary" / f"{method}_README.md",
    }


def build_examples(
    loss_rows: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    local_by_name: dict[str, dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
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
                raise KeyError(f"missing local error for {name}")
            terms.append((int(f(local, "layer")), local["module_type"], f(local, metric)))
        out.append(
            {
                "policy_id": row["policy_id"],
                "sample_kind": row.get("sample_kind", policy.get("sample_kind", "")),
                "selected_modules": int(f(row, "selected_modules")),
                "loss_delta_vs_dense": f(row, "loss_delta_vs_dense"),
                "terms": terms,
            }
        )
    if len(out) < 4:
        raise RuntimeError(f"need at least 4 loss samples, found {len(out)}")
    return out


def assign_splits(examples: list[dict[str, Any]], *, train_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    train_count = max(1, min(len(indices) - 1, round(len(indices) * train_ratio)))
    train = set(indices[:train_count])
    for idx, row in enumerate(examples):
        row["split"] = "train" if idx in train else "holdout"


def fit_model(examples: list[dict[str, Any]], *, method: str, metric: str, steps: int, lr: float, l2: float) -> dict[str, Any]:
    layer_index = {layer: idx for idx, layer in enumerate(LAYERS)}
    type_index = {typ: idx for idx, typ in enumerate(LINEAR_TYPES)}
    train_examples = [row for row in examples if row["split"] == "train"]
    x = design_tensor(train_examples, layer_index, type_index)
    y = torch.tensor([row["loss_delta_vs_dense"] for row in train_examples], dtype=torch.float64)
    log_layer = torch.zeros(len(LAYERS), dtype=torch.float64, requires_grad=True)
    log_type = torch.zeros(len(LINEAR_TYPES), dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(float(y.mean().item()), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_layer, log_type, bias], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        pred = predict_tensor(x, log_layer, log_type, bias)
        loss = torch.mean((pred - y) ** 2) + l2 * (torch.mean(log_layer**2) + torch.mean(log_type**2))
        loss.backward()
        opt.step()

    with torch.no_grad():
        type_centered = log_type - log_type.mean()
        layer_coef = torch.exp(log_layer)
        type_coef = torch.exp(type_centered)
        train_pred = predict_tensor(x, log_layer, log_type, bias)
        final_mse = float(torch.mean((train_pred - y) ** 2).item())
    return {
        "method": method,
        "formula": "bias + sum(kernel_local_error * layer_coef[layer] * type_coef[linear_type])",
        "local_error_metric": metric,
        "local_error_source": str(SOURCE_015_ROOT / "sensitivity" / "module_method_kernel_local_errors.csv"),
        "bias": float(bias.detach().item()),
        "train_mse": final_mse,
        "layer_coef": {str(layer): float(layer_coef[layer_index[layer]].item()) for layer in LAYERS},
        "type_coef": {typ: float(type_coef[type_index[typ]].item()) for typ in LINEAR_TYPES},
        "normalization": "type coefficients have geometric mean 1.0",
        "validity": "kernel_aware_real_runtime_forward_with_activation_quantization",
    }


def design_tensor(examples: list[dict[str, Any]], layer_index: dict[int, int], type_index: dict[str, int]) -> torch.Tensor:
    x = torch.zeros((len(examples), len(LAYERS), len(LINEAR_TYPES)), dtype=torch.float64)
    for row_idx, row in enumerate(examples):
        for layer, typ, error in row["terms"]:
            x[row_idx, layer_index[layer], type_index[typ]] += float(error)
    return x


def predict_tensor(x: torch.Tensor, log_layer: torch.Tensor, log_type: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    type_centered = log_type - log_type.mean()
    weights = torch.exp(log_layer)[:, None] * torch.exp(type_centered)[None, :]
    return bias + torch.sum(x * weights[None, :, :], dim=(1, 2))


def add_predictions(examples: list[dict[str, Any]], fit: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in examples:
        pred = fit["bias"]
        raw_error_sum = 0.0
        for layer, typ, error in row["terms"]:
            raw_error_sum += error
            pred += error * fit["layer_coef"][str(layer)] * fit["type_coef"][typ]
        out.append(
            {
                "policy_id": row["policy_id"],
                "split": row["split"],
                "sample_kind": row["sample_kind"],
                "selected_modules": row["selected_modules"],
                "loss_delta_vs_dense": row["loss_delta_vs_dense"],
                "pred_loss_delta": pred,
                "base_pred_loss_delta": pred,
                "raw_error_sum": raw_error_sum,
                "residual": row["loss_delta_vs_dense"] - pred,
            }
        )
    return sorted(out, key=lambda item: (item["split"], item["policy_id"]))


def fit_dense_calibration(rows: list[dict[str, Any]], *, l2: float) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train"]
    x = torch.tensor([calibration_features(row) for row in train_rows], dtype=torch.float64)
    y = torch.tensor([f(row, "loss_delta_vs_dense") for row in train_rows], dtype=torch.float64)
    penalty = torch.eye(x.shape[1], dtype=torch.float64) * l2
    penalty[0, 0] = 0.0
    coef = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
    return {
        "formula": "calibrated_pred = c0 + c1 * base_pred + c2 * base_pred^2 + c3 * log1p(selected_modules)",
        "l2": l2,
        "coefficients": {
            "intercept": float(coef[0].item()),
            "base_pred": float(coef[1].item()),
            "base_pred_squared": float(coef[2].item()),
            "log1p_selected_modules": float(coef[3].item()),
        },
    }


def apply_dense_calibration(rows: list[dict[str, Any]], calibration: dict[str, Any]) -> None:
    coef = calibration["coefficients"]
    weights = [
        coef["intercept"],
        coef["base_pred"],
        coef["base_pred_squared"],
        coef["log1p_selected_modules"],
    ]
    for row in rows:
        features = calibration_features(row)
        pred = sum(w * x for w, x in zip(weights, features))
        row["pred_loss_delta"] = pred
        row["calibrated_pred_loss_delta"] = pred
        row["residual"] = f(row, "loss_delta_vs_dense") - pred


def calibration_features(row: dict[str, Any]) -> list[float]:
    base_pred = f(row, "base_pred_loss_delta")
    return [1.0, base_pred, base_pred * base_pred, math.log1p(max(f(row, "selected_modules"), 0.0))]


def metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for split in ("train", "holdout", "all"):
        items = rows if split == "all" else [row for row in rows if row["split"] == split]
        if not items:
            continue
        y = [f(row, "loss_delta_vs_dense") for row in items]
        p = [f(row, "pred_loss_delta") for row in items]
        residual = [a - b for a, b in zip(y, p)]
        out.append(
            {
                "split": split,
                "rows": len(items),
                "pearson": pearson(p, y),
                "spearman": spearman(p, y),
                "mae": mean(abs(value) for value in residual),
                "rmse": math.sqrt(mean(value * value for value in residual)),
            }
        )
    return out


def plot_holdout(rows: list[dict[str, Any]], metrics: list[dict[str, Any]], path: Path, method: str) -> None:
    holdout = [row for row in rows if row["split"] == "holdout"]
    metric = next((row for row in metrics if row["split"] == "holdout"), {})
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter([f(row, "pred_loss_delta") for row in holdout], [f(row, "loss_delta_vs_dense") for row in holdout], alpha=0.8)
    if holdout:
        xs = [f(row, "pred_loss_delta") for row in holdout]
        ys = [f(row, "loss_delta_vs_dense") for row in holdout]
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel(f"Predicted {method} proxy loss delta")
    ax.set_ylabel("Measured loss delta vs dense")
    ax.set_title(
        f"Holdout {method} kernel proxy vs prefill loss"
        f"\nSpearman={f(metric, 'spearman', math.nan):.3f}, Pearson={f(metric, 'pearson', math.nan):.3f}, n={int(f(metric, 'rows'))}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_method_summary(path: Path, method: str, fit: dict[str, Any], metrics: list[dict[str, Any]], plot_path: Path) -> None:
    lines = [
        f"# {method} Kernel Precision Proxy Summary",
        "",
        f"Formula: `{fit['formula']}`",
        f"Local error metric: `{fit['local_error_metric']}`",
    ]
    calibration = fit.get("dense_calibration")
    if calibration:
        lines.extend(["", f"Dense calibration: `{calibration['formula']}`"])
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| split | rows | Pearson | Spearman | MAE | RMSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['split']} | {int(f(row, 'rows'))} | {f(row, 'pearson'):.4f} | "
            f"{f(row, 'spearman'):.4f} | {f(row, 'mae'):.6f} | {f(row, 'rmse'):.6f} |"
        )
    lines.extend(["", "## Main Plot", "", f"- `{plot_path}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_combined_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    path = output_root / "summary" / "kernel_nvfp4_proxy_summary.md"
    lines = [
        "# Kernel NVFP4 Proxy Summary",
        "",
        "| method | split | rows | Pearson | Spearman | MAE | RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['split']} | {int(f(row, 'rows'))} | {f(row, 'pearson'):.4f} | "
            f"{f(row, 'spearman'):.4f} | {f(row, 'mae'):.6f} | {f(row, 'rmse'):.6f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
