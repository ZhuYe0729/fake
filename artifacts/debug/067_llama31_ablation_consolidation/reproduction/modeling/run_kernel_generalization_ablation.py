#!/usr/bin/env python3
"""Compare direct fitting and Roofline residual modeling on unseen (N,K) groups."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
MODEL_ROOT = CUTLASS / "artifacts/modeling"
sys.path[:0] = [str(CUTLASS), str(CUTLASS / "modeling")]

from modeling.common import kernel_latency_model as common  # noqa: E402
from modeling.dense_bf16 import dense_bf16_model as dense_bf16  # noqa: E402
from modeling.dense_nvfp4 import dense_nvfp4_model  # noqa: E402
from modeling.marlin_nvfp4 import marlin_nvfp4_model  # noqa: E402
from modeling.sparse_bf16 import sparse_bf16_model as sparse_bf16  # noqa: E402
from modeling.sparse_nvfp4 import sparse_nvfp4_model  # noqa: E402


SEEDS = (1, 2, 3)
TEST_SIZE = 0.25
NEIGHBORS = 12
VARIANTS = ("direct_fit", "roofline", "roofline_shape_corrected")
QUANTIZED = {
    "dense_nvfp4": dense_nvfp4_model.CONFIG,
    "sparse_nvfp4": sparse_nvfp4_model.CONFIG,
    "marlin_nvfp4": marlin_nvfp4_model.CONFIG,
}


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    actual_rank = ranks(actual)
    predicted_rank = ranks(predicted)
    rank_denominator = np.linalg.norm(actual_rank - actual_rank.mean()) * np.linalg.norm(
        predicted_rank - predicted_rank.mean()
    )
    return {
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "mape": float(np.mean(np.abs(residual) / actual)),
        "p50_ape": float(np.quantile(np.abs(residual) / actual, 0.50)),
        "p90_ape": float(np.quantile(np.abs(residual) / actual, 0.90)),
        "spearman": (
            float(np.dot(actual_rank - actual_rank.mean(), predicted_rank - predicted_rank.mean()) / rank_denominator)
            if rank_denominator else 0.0
        ),
    }


def direct_fit_predict(train: pd.DataFrame, test: pd.DataFrame, make_features) -> np.ndarray:
    """Use the residual model's capacity but directly predict log latency."""
    train_features = make_features(train)
    test_features = make_features(test)[list(train_features.columns)]
    values = train_features.to_numpy(dtype=float)
    x_mean = values.mean(axis=0)
    x_scale = values.std(axis=0)
    x_scale[x_scale == 0.0] = 1.0
    train_x = (values - x_mean) / x_scale
    test_x = (test_features.to_numpy(dtype=float) - x_mean) / x_scale
    train_y = np.log(train["latency_ms"].to_numpy(dtype=float))

    design = np.column_stack([np.ones(len(train_x)), train_x])
    regularizer = np.eye(design.shape[1]) * 1e-3
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ train_y)

    distances = np.linalg.norm(test_x[:, None, :] - train_x[None, :, :], axis=2)
    neighbors = min(NEIGHBORS, len(train_x))
    indices = np.argpartition(distances, kth=neighbors - 1, axis=1)[:, :neighbors]
    local_distances = np.take_along_axis(distances, indices, axis=1)
    weights = 1.0 / (local_distances + 1e-6)
    local_log_latency = (weights * train_y[indices]).sum(axis=1) / weights.sum(axis=1)
    ridge_log_latency = np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients
    local_weight = np.clip(2.5 / (2.5 + local_distances.min(axis=1)), 0.25, 0.95)
    return np.exp(local_weight * local_log_latency + (1.0 - local_weight) * ridge_log_latency)


def select(data: pd.DataFrame, shapes: set[tuple[int, int, int]]) -> pd.DataFrame:
    keys = pd.MultiIndex.from_frame(data[["m", "n", "k"]])
    return data.loc[keys.isin(shapes)].reset_index(drop=True)


def load_kernel(kernel: str):
    if kernel == "dense_bf16":
        raw = dense_bf16.load_measurements(dense_bf16.DEFAULT_INPUT_ROOT)
        return (
            dense_bf16.aggregate_measurements(raw),
            dense_bf16.make_features,
            dense_bf16.split_shapes,
            lambda train: dense_bf16.train_model(train, neighbors=NEIGHBORS),
        )
    if kernel == "sparse_bf16":
        raw, skipped = sparse_bf16.load_measurements(sparse_bf16.DEFAULT_INPUT_ROOT)
        return (
            sparse_bf16.aggregate_measurements(raw),
            sparse_bf16.make_features,
            sparse_bf16.split_shapes,
            lambda train: sparse_bf16.train_model(train, skip_rows=skipped, neighbors=NEIGHBORS),
        )
    config = QUANTIZED[kernel]
    raw, skipped = common.load_measurements(config, common.DEFAULT_INPUT_ROOT)
    return (
        common.aggregate_measurements(config, raw),
        lambda frame: common.make_features(config, frame),
        common.split_shapes,
        lambda train: common.train_model(config, train, skip_rows=skipped, neighbors=NEIGHBORS),
    )


def saved_predictions(kernel: str) -> dict[tuple[int, int, int, int], dict[str, str]]:
    path = MODEL_ROOT / kernel / "validation_predictions.csv"
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split_mode"] == "nk"]
    return {(int(row["seed"]), int(row["m"]), int(row["n"]), int(row["k"])): row for row in rows}


def main() -> None:
    output_dir = OUT / "speed"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    validation_differences: list[float] = []
    for kernel in ("dense_bf16", "sparse_bf16", *QUANTIZED):
        data, make_features, split_shapes, train_residual = load_kernel(kernel)
        frozen = saved_predictions(kernel)
        for seed in SEEDS:
            train_shapes, test_shapes = split_shapes(data, "nk", TEST_SIZE, seed)
            train = select(data, train_shapes)
            test = select(data, test_shapes)
            direct = direct_fit_predict(train, test, make_features)
            residual_model = train_residual(train)
            predicted = residual_model.predict_frame(test[["m", "n", "k"]])
            predicted_by_shape = {
                (int(row.m), int(row.n), int(row.k)): row for row in predicted.itertuples(index=False)
            }
            baseline_field = "roofline_ms" if kernel == "dense_bf16" else "baseline_ms"
            for index, row in enumerate(test.itertuples(index=False)):
                shape = (int(row.m), int(row.n), int(row.k))
                model_row = predicted_by_shape[shape]
                saved = frozen[(seed, *shape)]
                validation_differences.append(
                    abs(float(model_row.latency_ms) - float(saved["latency_ms_pred"]))
                    / max(float(saved["latency_ms_pred"]), 1e-12)
                )
                actual = float(row.latency_ms)
                item: dict[str, object] = {
                    "kernel": kernel,
                    "seed": seed,
                    "m": shape[0],
                    "n": shape[1],
                    "k": shape[2],
                    "measured_ms": actual,
                    "predicted_direct_fit_ms": float(direct[index]),
                    "predicted_roofline_ms": float(getattr(model_row, baseline_field)),
                    "predicted_roofline_shape_corrected_ms": float(model_row.latency_ms),
                }
                for variant in VARIANTS:
                    item[f"ape_{variant}"] = abs(actual - float(item[f"predicted_{variant}_ms"])) / actual
                rows.append(item)

    with (output_dir / "kernel_generalization_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_rows: list[dict[str, object]] = []
    for kernel in ("all", "dense_bf16", "sparse_bf16", *QUANTIZED):
        selected_rows = rows if kernel == "all" else [row for row in rows if row["kernel"] == kernel]
        actual = np.array([float(row["measured_ms"]) for row in selected_rows])
        for variant in VARIANTS:
            predicted = np.array([float(row[f"predicted_{variant}_ms"]) for row in selected_rows])
            metric_rows.append({
                "kernel": kernel,
                "variant": variant,
                "samples": len(selected_rows),
                **metrics(actual, predicted),
            })
    with (output_dir / "kernel_generalization_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    validation = {
        "split_mode": "nk",
        "seeds": list(SEEDS),
        "test_size": TEST_SIZE,
        "neighbors": NEIGHBORS,
        "kernels": ["dense_bf16", "sparse_bf16", *QUANTIZED],
        "rows": len(rows),
        "max_relative_difference_vs_saved_residual_prediction": max(validation_differences),
        "reproduces_saved_predictions_at_1e-8": max(validation_differences) <= 1e-8,
    }
    (output_dir / "kernel_generalization_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if not validation["reproduces_saved_predictions_at_1e-8"]:
        raise RuntimeError("unseen-(N,K) evaluation does not reproduce the saved model validation")


if __name__ == "__main__":
    main()
