#!/usr/bin/env python3
"""Fit a phase-aware local-error proxy to canonical real-vLLM NLL labels."""
from __future__ import annotations

import csv
import json
import math
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scenario import EXP
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
PHASES = ("prefill", "decode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--report-name", default="quality")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rank(values: list[float]) -> list[float]:
    return [1 + sum(other < value for other in values) + .5 * sum(other == value for other in values)
            for value in values]


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    residuals = [a - p for a, p in zip(actual, predicted)]
    ra, rp = rank(actual), rank(predicted)
    ma, mp = sum(ra) / len(ra), sum(rp) / len(rp)
    denominator = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((x - mp) ** 2 for x in rp))
    return {"mae": sum(abs(x) for x in residuals) / len(residuals),
            "rmse": math.sqrt(sum(x * x for x in residuals) / len(residuals)),
            "mean_signed_error": sum(residuals) / len(residuals),
            "spearman": sum((x - ma) * (y - mp) for x, y in zip(ra, rp)) / denominator if denominator else 0.0}


def local_errors() -> dict[tuple[str, int, str, str], float]:
    result = {}
    for phase in PHASES:
        for method in METHODS:
            if method == "dense_bf16":
                for bucket in range(4):
                    for typ in TYPES:
                        result[phase, bucket, typ, method] = 0.0
                continue
            for row in read_csv(EXP / "local_errors" / f"{phase}_{method}.csv"):
                result[phase, int(row["layer_bucket"]), row["fused_type"], method] = float(row["output_rel_mse"])
    expected = len(PHASES) * len(METHODS) * 4 * len(TYPES)
    if len(result) != expected:
        raise RuntimeError(f"incomplete local-error table: {len(result)} vs {expected}")
    return result


def features(policy: dict, errors: dict[tuple[str, int, str, str], float]) -> torch.Tensor:
    value = torch.zeros((2, len(METHODS), 4, len(TYPES)), dtype=torch.float64)
    for name, methods in policy["method_map"].items():
        layer_bucket = int(name.split(".")[2]) // 8
        raw_type = name.rsplit(".", 1)[-1]
        typ = "qkv_proj" if raw_type in {"q_proj", "k_proj", "v_proj"} else "gate_up_proj" if raw_type in {"gate_proj", "up_proj"} else raw_type
        for phase_index, phase in enumerate(PHASES):
            method = methods[f"{phase}_method"]
            value[phase_index, METHODS.index(method), layer_bucket, TYPES.index(typ)] += errors[phase, layer_bucket, typ, method]
    return value


def main() -> None:
    args = parse_args()
    manifest = json.loads((EXP / "policies/prefill_decode/manifest.json").read_text())
    if args.split_json:
        split = json.loads(args.split_json.read_text())
        holdout = set(split["holdout"])
        split_of = lambda item: "holdout" if item["policy_id"] in holdout else "train"
        split_description = split["selection"]
    else:
        split_of = lambda item: item["split"]
        split_description = "legacy p54-p71 narrow balanced holdout"
    labels = {row["policy_id"]: row for row in read_csv(EXP / "calibration/nll/prefill_decode.csv")}
    policies = [json.loads(Path(row["path"]).read_text()) for row in manifest]
    errors = local_errors()
    x = torch.stack([features(policy, errors) for policy in policies])
    y = torch.tensor([float(labels[row["policy_id"]]["target_delta_nll"]) for row in manifest], dtype=torch.float64)
    train = torch.tensor([split_of(row) == "train" for row in manifest])
    scale = x[train].sum((1, 2, 3, 4)).mean().clamp(min=1e-12)
    x = x / scale

    # Positive, low-dimensional calibration: phase-specific behavior without
    # memorizing policies or treating compression methods as interchangeable.
    global_factor = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    phase_factor = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    method_factor = torch.zeros(len(METHODS), dtype=torch.float64, requires_grad=True)
    bucket_factor = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    type_factor = torch.zeros(len(TYPES), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    parameters = [global_factor, phase_factor, method_factor, bucket_factor, type_factor, bias]
    optimizer = torch.optim.Adam(parameters, lr=0.03)
    for _ in range(4000):
        optimizer.zero_grad()
        coefficient = torch.relu(global_factor + phase_factor[:, None, None, None] + method_factor[None, :, None, None] + bucket_factor[None, None, :, None] + type_factor[None, None, None, :])
        prediction = bias + (x * coefficient).sum((1, 2, 3, 4))
        loss = ((prediction[train] - y[train]) ** 2).mean() + .05 * sum((item * item).mean() for item in parameters[:-1])
        loss.backward(); optimizer.step()
    with torch.no_grad():
        coefficient = torch.relu(global_factor + phase_factor[:, None, None, None] + method_factor[None, :, None, None] + bucket_factor[None, None, :, None] + type_factor[None, None, None, :])
        prediction = (bias + (x * coefficient).sum((1, 2, 3, 4))).tolist()

    rows = []
    for index, item in enumerate(manifest):
        rows.append({"policy_id": item["policy_id"], "split": split_of(item), "policy_kind": item["policy_kind"],
                     "actual_delta_nll": float(y[index]), "predicted_delta_nll": prediction[index],
                     "residual_actual_minus_predicted": float(y[index]) - prediction[index]})
    report = EXP / "reports" / args.report_name; report.mkdir(parents=True, exist_ok=True)
    with (report / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"proxy": "phase-aware positive local-error aggregation: global + phase + method + layer-bucket + fused-type ReLU calibration",
               "label_backend": "canonical sparse phase_hetero real-vLLM teacher-forced prefill-decode NLL",
               "feature_backend": "same compression wrappers as phase runtime; canonical sparse weights; prefill/decode local output_rel_mse",
               "fit_split": "54 train / 18 holdout; " + split_description,
               "metrics": {split: metrics([row["actual_delta_nll"] for row in rows if row["split"] == split],
                                            [row["predicted_delta_nll"] for row in rows if row["split"] == split])
                           for split in ("train", "holdout")}}
    (report / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    model = {"feature_scale": float(scale), "global": float(global_factor.detach()), "phase": phase_factor.detach().tolist(),
             "method": method_factor.detach().tolist(), "bucket": bucket_factor.detach().tolist(), "type": type_factor.detach().tolist(),
             "bias": float(bias.detach()), "coefficients": coefficient.detach().tolist()}
    (report / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    for split, color, marker in (("train", "#4c78a8", "o"), ("holdout", "#e45756", "s")):
        subset = [row for row in rows if row["split"] == split]
        plt.scatter([row["actual_delta_nll"] for row in subset], [row["predicted_delta_nll"] for row in subset], color=color, marker=marker, label=split)
    upper = max(max(row[key] for row in rows) for key in ("actual_delta_nll", "predicted_delta_nll"))
    lower = min(min(row[key] for row in rows) for key in ("actual_delta_nll", "predicted_delta_nll"))
    plt.plot([lower, upper], [lower, upper], "--", color="#555555"); plt.xlabel("Measured real-vLLM ΔNLL"); plt.ylabel("Predicted ΔNLL"); plt.legend(); plt.tight_layout(); plt.savefig(report / "predicted_vs_measured.png", dpi=180); plt.close()
    lines = ["# Llama3.1 canonical prefill-decode quality model", "", "| split | MAE | RMSE | Spearman |", "|---|---:|---:|---:|"]
    lines += [f"| {split} | {summary['metrics'][split]['mae']:.6f} | {summary['metrics'][split]['rmse']:.6f} | {summary['metrics'][split]['spearman']:.4f} |" for split in ("train", "holdout")]
    (report / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
