#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from common_sparse_bf16_proxy import DEBUG_ROOT, f, read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate proxy signal after removing count/raw-local baseline.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--ablation-subdir", default="stratified_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root / args.ablation_subdir
    rows = read_csv(root / "proxy_ablation_predictions.csv")
    out_rows = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        baseline = fit_baseline([row for row in method_rows if row["split"] == "train"])
        for variant in sorted({row["variant"] for row in method_rows}):
            items = [row for row in method_rows if row["variant"] == variant and row["split"] == "holdout"]
            if not items:
                continue
            y = np.array([f(row, "loss_delta_vs_dense") for row in items], dtype=np.float64)
            pred = np.array([f(row, "pred_loss_delta") for row in items], dtype=np.float64)
            base = np.array([baseline_predict(baseline, row) for row in items], dtype=np.float64)
            y_resid = y - base
            pred_resid = pred - base
            out_rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "rows": len(items),
                    "raw_count_baseline_rmse": rmse(y, base),
                    "proxy_rmse": rmse(y, pred),
                    "residual_pearson": pearson(pred_resid.tolist(), y_resid.tolist()),
                    "residual_spearman": spearman(pred_resid.tolist(), y_resid.tolist()),
                    "residual_mae": mean(abs(a - b) for a, b in zip(y_resid, pred_resid)),
                    "residual_rmse": rmse(y_resid, pred_resid),
                }
            )
    write_csv(root / "stratified_residual_proxy_metrics.csv", out_rows)
    write_report(root / "stratified_residual_proxy_summary.md", out_rows)
    print(f"wrote {root / 'stratified_residual_proxy_summary.md'}")


def fit_baseline(rows: list[dict[str, Any]]) -> np.ndarray:
    x = np.array([[1.0, f(row, "selected_modules"), f(row, "raw_error_sum")] for row in rows], dtype=np.float64)
    y = np.array([f(row, "loss_delta_vs_dense") for row in rows], dtype=np.float64)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def baseline_predict(coef: np.ndarray, row: dict[str, Any]) -> float:
    return float(np.array([1.0, f(row, "selected_modules"), f(row, "raw_error_sum")], dtype=np.float64) @ coef)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stratified Residual Proxy Summary",
        "",
        "Residual metrics subtract a train-fitted baseline using only `selected_modules` and `raw_error_sum`.",
        "",
        "| method | variant | rows | baseline RMSE | proxy RMSE | residual Pearson | residual Spearman | residual MAE | residual RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['variant']} | {int(f(row, 'rows'))} | "
            f"{f(row, 'raw_count_baseline_rmse'):.6f} | {f(row, 'proxy_rmse'):.6f} | "
            f"{f(row, 'residual_pearson'):.4f} | {f(row, 'residual_spearman'):.4f} | "
            f"{f(row, 'residual_mae'):.6f} | {f(row, 'residual_rmse'):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rmse(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.sqrt(np.mean((xs - ys) ** 2)))


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
