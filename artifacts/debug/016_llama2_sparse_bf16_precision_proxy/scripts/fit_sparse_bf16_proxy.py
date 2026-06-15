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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit multiplicative sparse BF16 local-error proxy.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-014-root", type=Path, default=SOURCE_014_ROOT)
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = policy_paths(args.output_root)
    policies = {row["policy_id"]: row for row in read_csv(paths["policies"])}
    loss_rows = read_csv(paths["loss"])
    local_rows = load_sparse_local_errors(args.source_014_root)
    local_by_name = {row["module_name"]: row for row in local_rows}
    examples = build_examples(loss_rows, policies, local_by_name, args.metric)
    assign_splits(examples, train_ratio=args.train_ratio, seed=args.seed)
    fit = fit_model(examples, metric=args.metric, steps=args.steps, lr=args.lr, l2=args.l2)
    predictions = add_predictions(examples, fit)
    metrics = metric_rows(predictions)

    write_csv(paths["predictions"], predictions)
    write_csv(paths["metrics"], metrics)
    write_json(paths["model"], fit)
    plot_holdout(predictions, metrics, paths["plot"])
    write_summary(paths["summary"], fit, metrics, paths["plot"])
    print(f"wrote model to {paths['model']}")
    print(f"wrote holdout plot to {paths['plot']}")


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


def fit_model(examples: list[dict[str, Any]], *, metric: str, steps: int, lr: float, l2: float) -> dict[str, Any]:
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
        "formula": "bias + sum(local_error * layer_coef[layer] * type_coef[linear_type])",
        "local_error_metric": metric,
        "bias": float(bias.detach().item()),
        "train_mse": final_mse,
        "layer_coef": {str(layer): float(layer_coef[layer_index[layer]].item()) for layer in LAYERS},
        "type_coef": {typ: float(type_coef[type_index[typ]].item()) for typ in LINEAR_TYPES},
        "normalization": "type coefficients have geometric mean 1.0",
    }


def design_tensor(
    examples: list[dict[str, Any]],
    layer_index: dict[int, int],
    type_index: dict[str, int],
) -> torch.Tensor:
    x = torch.zeros((len(examples), len(LAYERS), len(LINEAR_TYPES)), dtype=torch.float64)
    for row_idx, row in enumerate(examples):
        for layer, typ, error in row["terms"]:
            x[row_idx, layer_index[layer], type_index[typ]] += float(error)
    return x


def predict_tensor(
    x: torch.Tensor,
    log_layer: torch.Tensor,
    log_type: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
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
                "raw_error_sum": raw_error_sum,
                "residual": row["loss_delta_vs_dense"] - pred,
            }
        )
    return sorted(out, key=lambda item: (item["split"], item["policy_id"]))


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


def plot_holdout(rows: list[dict[str, Any]], metrics: list[dict[str, Any]], path: Path) -> None:
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
    ax.set_xlabel("Predicted sparse BF16 proxy loss delta")
    ax.set_ylabel("Measured loss delta vs dense")
    ax.set_title(
        "Holdout sparse BF16 proxy vs prefill loss"
        f"\nSpearman={f(metric, 'spearman', math.nan):.3f}, Pearson={f(metric, 'pearson', math.nan):.3f}, n={int(f(metric, 'rows'))}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, fit: dict[str, Any], metrics: list[dict[str, Any]], plot_path: Path) -> None:
    lines = [
        "# Sparse BF16 Precision Proxy Summary",
        "",
        f"Formula: `{fit['formula']}`",
        f"Local error metric: `{fit['local_error_metric']}`",
        "",
        "## Metrics",
        "",
        "| split | rows | Pearson | Spearman | MAE | RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['split']} | {int(f(row, 'rows'))} | {f(row, 'pearson'):.4f} | "
            f"{f(row, 'spearman'):.4f} | {f(row, 'mae'):.6f} | {f(row, 'rmse'):.6f} |"
        )
    lines.extend(["", "## Main Plot", "", f"- `{plot_path}`", ""])
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
