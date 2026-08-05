#!/usr/bin/env python3
"""Fit the compact three-stage quality-model ablation on frozen 057 labels."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
PHASES = ("prefill", "decode")
VARIANTS = ("local_sum", "structure_aware", "full_phase_aware")


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
        "mae": sum(abs(value) for value in residuals) / len(residuals),
        "rmse": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "mean_signed_error": sum(residuals) / len(residuals),
        "spearman": (
            sum((x - actual_mean) * (y - predicted_mean) for x, y in zip(actual_rank, predicted_rank))
            / denominator
            if denominator else 0.0
        ),
    }


def load_local_errors() -> dict[tuple[str, int, str, str], float]:
    result: dict[tuple[str, int, str, str], float] = {}
    for phase in PHASES:
        for bucket in range(4):
            for module_type in TYPES:
                result[phase, bucket, module_type, "dense_bf16"] = 0.0
        for method in METHODS[1:]:
            path = SOURCE / "local_errors" / f"{phase}_{method}.csv"
            for row in read_csv(path):
                result[phase, int(row["layer_bucket"]), row["fused_type"], method] = float(row["output_rel_mse"])
    expected = len(PHASES) * len(METHODS) * 4 * len(TYPES)
    if len(result) != expected:
        raise RuntimeError(f"incomplete local-error table: {len(result)} != {expected}")
    return result


def policy_features(policy: dict, errors: dict[tuple[str, int, str, str], float]) -> torch.Tensor:
    value = torch.zeros((len(PHASES), len(METHODS), 4, len(TYPES)), dtype=torch.float64)
    for name, selected in policy["method_map"].items():
        bucket = int(name.split(".")[2]) // 8
        raw_type = name.rsplit(".", 1)[-1]
        module_type = (
            "qkv_proj" if raw_type in {"q_proj", "k_proj", "v_proj", "qkv_proj"}
            else "gate_up_proj" if raw_type in {"gate_proj", "up_proj", "gate_up_proj"}
            else raw_type
        )
        for phase_index, phase in enumerate(PHASES):
            method = selected[f"{phase}_method"]
            value[phase_index, METHODS.index(method), bucket, TYPES.index(module_type)] += errors[
                phase, bucket, module_type, method
            ]
    return value


def fit_variant(name: str, x: torch.Tensor, y: torch.Tensor, train: torch.Tensor) -> tuple[torch.Tensor, dict]:
    global_factor = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    parameters: list[torch.Tensor] = [global_factor]
    model: dict[str, torch.Tensor] = {"global": global_factor}

    if name in {"structure_aware", "full_phase_aware"}:
        model["method"] = torch.zeros(len(METHODS), dtype=torch.float64, requires_grad=True)
        model["bucket"] = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        model["type"] = torch.zeros(len(TYPES), dtype=torch.float64, requires_grad=True)
        parameters.extend([model["method"], model["bucket"], model["type"]])
    if name == "full_phase_aware":
        model["phase"] = torch.zeros(len(PHASES), dtype=torch.float64, requires_grad=True)
        parameters.insert(1, model["phase"])
    parameters.append(bias)

    def predict() -> tuple[torch.Tensor, torch.Tensor]:
        if name == "local_sum":
            coefficient = torch.relu(global_factor)
        elif name == "structure_aware":
            coefficient = torch.relu(
                global_factor
                + model["method"][:, None, None]
                + model["bucket"][None, :, None]
                + model["type"][None, None, :]
            )
            coefficient = coefficient.unsqueeze(0)
        else:
            coefficient = torch.relu(
                global_factor
                + model["phase"][:, None, None, None]
                + model["method"][None, :, None, None]
                + model["bucket"][None, None, :, None]
                + model["type"][None, None, None, :]
            )
        return bias + (x * coefficient).sum((1, 2, 3, 4)), coefficient

    optimizer = torch.optim.Adam(parameters, lr=0.03)
    for _ in range(4000):
        optimizer.zero_grad()
        prediction, _ = predict()
        loss = ((prediction[train] - y[train]) ** 2).mean()
        loss = loss + 0.05 * sum((parameter * parameter).mean() for parameter in parameters[:-1])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction, coefficient = predict()
    serialized = {key: value.detach().tolist() for key, value in model.items()}
    serialized.update({"bias": float(bias.detach()), "coefficients": coefficient.detach().tolist()})
    return prediction, serialized


def main() -> None:
    output_dir = OUT / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "policies/prefill_decode/manifest.json").read_text())
    split = json.loads((SOURCE / "policies/prefill_decode/coverage_holdout.json").read_text())
    train_ids, holdout_ids = set(split["train"]), set(split["holdout"])
    if len(train_ids) != 54 or len(holdout_ids) != 18 or train_ids & holdout_ids:
        raise RuntimeError("quality split must be a disjoint fixed 54/18 split")
    labels = {row["policy_id"]: float(row["target_delta_nll"])
              for row in read_csv(SOURCE / "nll/prefill_decode.csv")}
    errors = load_local_errors()
    policies = [json.loads(Path(row["path"]).read_text()) for row in manifest]
    x = torch.stack([policy_features(policy, errors) for policy in policies])
    y = torch.tensor([labels[row["policy_id"]] for row in manifest], dtype=torch.float64)
    train = torch.tensor([row["policy_id"] in train_ids for row in manifest])
    feature_scale = x[train].sum((1, 2, 3, 4)).mean().clamp(min=1e-12)
    x = x / feature_scale

    metric_rows: list[dict[str, object]] = []
    predictions_by_variant: dict[str, list[float]] = {}
    for variant in VARIANTS:
        prediction, model = fit_variant(variant, x, y, train)
        predicted = prediction.tolist()
        predictions_by_variant[variant] = predicted
        model["feature_scale"] = float(feature_scale)
        model["variant"] = variant
        (output_dir / f"model_{variant}.json").write_text(json.dumps(model, indent=2) + "\n")
        for split_name, ids in (("train", train_ids), ("holdout", holdout_ids)):
            indices = [index for index, row in enumerate(manifest) if row["policy_id"] in ids]
            values = metrics([float(y[index]) for index in indices], [predicted[index] for index in indices])
            metric_rows.append({"variant": variant, "split": split_name, **values})

    prediction_rows = []
    for index, row in enumerate(manifest):
        actual = float(y[index])
        split_name = "train" if row["policy_id"] in train_ids else "holdout"
        item: dict[str, object] = {
            "policy_id": row["policy_id"], "split": split_name,
            "policy_kind": row["policy_kind"], "measured_delta_nll": actual,
        }
        for variant in VARIANTS:
            predicted = predictions_by_variant[variant][index]
            item[f"predicted_{variant}"] = predicted
            item[f"residual_{variant}"] = actual - predicted
        prediction_rows.append(item)

    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)

    frozen = {row["policy_id"]: float(row["predicted_delta_nll"])
              for row in read_csv(SOURCE / "reports/quality_coverage_holdout/predictions.csv")}
    differences = [abs(predictions_by_variant["full_phase_aware"][index] - frozen[row["policy_id"]])
                   for index, row in enumerate(manifest)]
    validation = {
        "source": str(SOURCE), "quality_train": sorted(train_ids), "quality_holdout": sorted(holdout_ids),
        "full_vs_frozen_max_abs_difference": max(differences),
        "full_matches_frozen_at_1e-8": max(differences) <= 1e-8,
    }
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if not validation["full_matches_frozen_at_1e-8"]:
        raise RuntimeError(f"full quality model does not reproduce frozen 057: {max(differences)}")


if __name__ == "__main__":
    main()

