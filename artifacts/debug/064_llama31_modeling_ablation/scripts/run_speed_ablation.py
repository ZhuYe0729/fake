#!/usr/bin/env python3
"""Evaluate the compact three-stage latency ablation on frozen 057 E2E labels."""
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
SOURCE = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
sys.path[:0] = [str(CUTLASS), str(CUTLASS / "modeling")]
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402

METHOD_TO_KERNEL = {"w4a16_ours": "marlin_nvfp4"}
DECODE_STEPS = 63
VARIANTS = ("roofline", "shape_corrected", "roofline_policy_calibrated", "policy_calibrated")
LOCAL_PROFILE = ROOT / "artifacts/debug/059_llama31_prefill_speed_decomposition/exact_micro/targeted_profile.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ranks(values: list[float]) -> list[float]:
    return [1 + sum(other < value for other in values) + 0.5 * sum(other == value for other in values)
            for value in values]


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    residuals = [a - p for a, p in zip(actual, predicted)]
    actual_rank, predicted_rank = ranks(actual), ranks(predicted)
    actual_mean = sum(actual_rank) / len(actual_rank)
    predicted_mean = sum(predicted_rank) / len(predicted_rank)
    denominator = math.sqrt(
        sum((value - actual_mean) ** 2 for value in actual_rank)
        * sum((value - predicted_mean) ** 2 for value in predicted_rank)
    )
    return {
        "mae_ms": sum(abs(value) for value in residuals) / len(residuals),
        "rmse_ms": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "mape": sum(abs(a - p) / a for a, p in zip(actual, predicted)) / len(actual),
        "mean_signed_error_ms": sum(residuals) / len(residuals),
        "spearman": (
            sum((x - actual_mean) * (y - predicted_mean) for x, y in zip(actual_rank, predicted_rank))
            / denominator
            if denominator else 0.0
        ),
    }


def pava(values: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            blocks[-1][0] += right[0]
            blocks[-1][1] += right[1]
    return [total / count for total, count in blocks for _ in range(int(count))]


def monotone_predict(train: list[tuple[float, float]], value: float) -> float:
    pairs = sorted(train)
    xs = [x for x, _ in pairs]
    ys = pava([y for _, y in pairs])
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for index in range(1, len(xs)):
        if value <= xs[index]:
            ratio = (value - xs[index - 1]) / (xs[index] - xs[index - 1])
            return ys[index - 1] + ratio * (ys[index] - ys[index - 1])
    raise AssertionError("unreachable")


def positive_affine_fit(train: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit the lowest-capacity E2E overhead map while preserving policy order."""
    pairs = sorted(train)
    design = np.column_stack([np.ones(len(pairs)), [x for x, _ in pairs]])
    target = np.array([y for _, y in pairs], dtype=float)
    intercept, slope = np.linalg.lstsq(design, target, rcond=None)[0]
    if slope <= 0.0:
        raise RuntimeError(f"E2E affine calibration must preserve ordering, got slope={slope}")
    return float(intercept), float(slope)


def kernel_costs() -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str, str], float], float]:
    actions = read_csv(SOURCE / "speed/action_support.csv")
    predictor = KernelLatencyPredictor(model_root=DEFAULT_MODEL_ROOT)
    roofline: dict[tuple[str, str, str], float] = {}
    corrected: dict[tuple[str, str, str], float] = {}
    max_relative_difference = 0.0
    for kernel, model in predictor.models.items():
        kernel_rows = [row for row in actions if row["kernel"] == kernel and row["supported"] == "True"]
        shapes = pd.DataFrame([{"m": int(row["m"]), "n": int(row["n"]), "k": int(row["k"])}
                               for row in kernel_rows]).drop_duplicates()
        predicted = model.predict_frame(shapes)
        by_shape = {(int(row["m"]), int(row["n"]), int(row["k"])): row
                    for row in predicted.to_dict("records")}
        for row in kernel_rows:
            shape = (int(row["m"]), int(row["n"]), int(row["k"]))
            item = by_shape[shape]
            baseline_key = "roofline_ms" if kernel == "dense_bf16" else "baseline_ms"
            key = (row["phase"], row["module_name"], kernel)
            roofline[key] = float(item[baseline_key])
            corrected[key] = float(row["latency_ms"])
            current = float(item["latency_ms"])
            frozen = corrected[key]
            max_relative_difference = max(max_relative_difference, abs(current - frozen) / max(frozen, 1e-12))
    return roofline, corrected, max_relative_difference


def evaluate_targeted_local(predictor: KernelLatencyPredictor, output_dir: Path) -> dict[str, dict[str, float]]:
    """Compare baseline/residual models on independent, non-exact Llama3 module profiles."""
    source_rows = [row for row in read_csv(LOCAL_PROFILE)
                   if row["benchmark_level"] == "module_forward" and row["status"] == "pass"]
    rows: list[dict[str, object]] = []
    for kernel, model in predictor.models.items():
        selected = [row for row in source_rows if row["kernel"] == kernel]
        shapes = pd.DataFrame([{"m": int(row["m"]), "n": int(row["n"]), "k": int(row["k"])}
                               for row in selected])
        predictions = model.predict_frame(shapes)
        by_shape = {(int(row["m"]), int(row["n"]), int(row["k"])): row
                    for row in predictions.to_dict("records")}
        for row in selected:
            shape = (int(row["m"]), int(row["n"]), int(row["k"]))
            prediction = by_shape[shape]
            # Exact lookup is a useful production feature, but it is excluded
            # here so the residual ablation measures prediction rather than
            # retrieval of a training shape.
            if prediction["source"] == "measured":
                continue
            baseline_key = "roofline_ms" if kernel == "dense_bf16" else "baseline_ms"
            actual = float(row["latency_ms"])
            baseline = float(prediction[baseline_key])
            corrected = float(prediction["latency_ms"])
            rows.append({
                "kernel": kernel, "m": shape[0], "n": shape[1], "k": shape[2],
                "measured_ms": actual, "predicted_roofline_ms": baseline,
                "predicted_shape_corrected_ms": corrected,
                "ape_roofline": abs(actual - baseline) / actual,
                "ape_shape_corrected": abs(actual - corrected) / actual,
            })
    if not rows:
        raise RuntimeError("no independent targeted local profiles were available")
    with (output_dir / "local_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {}
    metric_rows = []
    actual = [float(row["measured_ms"]) for row in rows]
    for variant in ("roofline", "shape_corrected"):
        predicted = [float(row[f"predicted_{variant}_ms"]) for row in rows]
        result[variant] = metrics(actual, predicted)
        metric_rows.append({"variant": variant, "samples": len(rows), **result[variant]})
    with (output_dir / "local_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    return result


def aggregate(policy: dict, costs: dict[tuple[str, str, str], float]) -> float:
    prefill = 0.0
    decode = 0.0
    for module_name, methods in policy["method_map"].items():
        prefill_kernel = METHOD_TO_KERNEL.get(methods["prefill_method"], methods["prefill_method"])
        decode_kernel = METHOD_TO_KERNEL.get(methods["decode_method"], methods["decode_method"])
        prefill += costs["prefill", module_name, prefill_kernel]
        decode += costs["decode", module_name, decode_kernel]
    return prefill + DECODE_STEPS * decode


def main() -> None:
    output_dir = OUT / "speed"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((SOURCE / "speed/calibration/metadata.json").read_text())
    train_ids = set(metadata["train_policies"])
    holdout_ids = set(metadata["holdout_policies"])
    if len(train_ids) != 7 or len(holdout_ids) != 5 or train_ids & holdout_ids:
        raise RuntimeError("speed split must be a disjoint fixed 7/5 split")
    frozen_rows = read_csv(SOURCE / "speed/calibration/calibration.csv")
    measured = {row["policy_id"]: float(row["e2e_median_ms"]) for row in frozen_rows}
    frozen_raw = {row["policy_id"]: float(row["raw_predicted_linear_ms"]) for row in frozen_rows}
    frozen_calibrated = {row["policy_id"]: float(row["monotone_prediction_ms"]) for row in frozen_rows}
    policy_dir = SOURCE / "speed/calibration/policies"

    roofline_cost, corrected_cost, current_vs_frozen = kernel_costs()
    predictor = KernelLatencyPredictor(model_root=DEFAULT_MODEL_ROOT)
    local_result = evaluate_targeted_local(predictor, output_dir)
    raw_roofline: dict[str, float] = {}
    raw_corrected: dict[str, float] = {}
    for policy_id in measured:
        policy = json.loads((policy_dir / f"{policy_id}.json").read_text())
        raw_roofline[policy_id] = aggregate(policy, roofline_cost)
        raw_corrected[policy_id] = aggregate(policy, corrected_cost)

    dense_id = "p00"
    roofline_scale = measured[dense_id] / raw_roofline[dense_id]
    corrected_scale = measured[dense_id] / raw_corrected[dense_id]
    monotone_train = [(raw_corrected[policy_id], measured[policy_id]) for policy_id in sorted(train_ids)]
    affine_intercept, affine_slope = positive_affine_fit(monotone_train)
    roofline_affine_intercept, roofline_affine_slope = positive_affine_fit(
        [(raw_roofline[policy_id], measured[policy_id]) for policy_id in sorted(train_ids)]
    )
    pava_predictions = {
        policy_id: monotone_predict(monotone_train, value) for policy_id, value in raw_corrected.items()
    }
    predictions = {
        "roofline": {policy_id: value * roofline_scale for policy_id, value in raw_roofline.items()},
        "shape_corrected": {policy_id: value * corrected_scale for policy_id, value in raw_corrected.items()},
        "roofline_policy_calibrated": {
            policy_id: roofline_affine_intercept + roofline_affine_slope * value
            for policy_id, value in raw_roofline.items()
        },
        "policy_calibrated": {
            policy_id: affine_intercept + affine_slope * value for policy_id, value in raw_corrected.items()
        },
    }

    rows = []
    for policy_id in measured:
        item: dict[str, object] = {
            "policy_id": policy_id,
            "split": "train" if policy_id in train_ids else "holdout",
            "measured_e2e_ms": measured[policy_id],
            "raw_roofline_ms": raw_roofline[policy_id],
            "raw_shape_corrected_ms": raw_corrected[policy_id],
            "predicted_policy_calibrated_pava_ms": pava_predictions[policy_id],
        }
        for variant in VARIANTS:
            item[f"predicted_{variant}_ms"] = predictions[variant][policy_id]
            item[f"residual_{variant}_ms"] = measured[policy_id] - predictions[variant][policy_id]
        rows.append(item)
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    metric_rows = []
    metric_predictions = {**predictions, "policy_calibrated_pava": pava_predictions}
    for variant, variant_predictions in metric_predictions.items():
        for split_name, ids in (("train", train_ids), ("holdout", holdout_ids)):
            actual = [measured[row["policy_id"]] for row in frozen_rows if row["policy_id"] in ids]
            predicted = [variant_predictions[row["policy_id"]] for row in frozen_rows if row["policy_id"] in ids]
            metric_rows.append({"variant": variant, "split": split_name, **metrics(actual, predicted)})
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)

    raw_difference = max(abs(raw_corrected[policy_id] - frozen_raw[policy_id]) for policy_id in measured)
    pava_difference = max(abs(pava_predictions[policy_id] - frozen_calibrated[policy_id])
                          for policy_id in measured)
    validation = {
        "source": str(SOURCE), "speed_train": sorted(train_ids), "speed_holdout": sorted(holdout_ids),
        "current_kernel_model_vs_frozen_action_max_relative_difference": current_vs_frozen,
        "shape_corrected_raw_vs_frozen_max_abs_difference_ms": raw_difference,
        "pava_diagnostic_vs_frozen_max_abs_difference_ms": pava_difference,
        "shape_corrected_matches_frozen_at_1e-8": raw_difference <= 1e-8,
        "pava_diagnostic_matches_frozen_at_1e-8": pava_difference <= 1e-8,
        "roofline_dense_anchor_scale": roofline_scale,
        "shape_corrected_dense_anchor_scale": corrected_scale,
        "policy_affine_intercept_ms": affine_intercept,
        "policy_affine_slope": affine_slope,
        "policy_affine_preserves_raw_order": affine_slope > 0.0,
        "roofline_policy_affine_intercept_ms": roofline_affine_intercept,
        "roofline_policy_affine_slope": roofline_affine_slope,
        "targeted_local_profile": str(LOCAL_PROFILE),
        "targeted_local_shape_correction_improves_mape": (
            local_result["shape_corrected"]["mape"] < local_result["roofline"]["mape"]
        ),
    }
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if not validation["shape_corrected_matches_frozen_at_1e-8"] or not validation["pava_diagnostic_matches_frozen_at_1e-8"]:
        raise RuntimeError("speed ablation does not reproduce the frozen 057 predictor")
    if not validation["targeted_local_shape_correction_improves_mape"]:
        raise RuntimeError("shape correction did not improve independent targeted local MAPE")


if __name__ == "__main__":
    main()
