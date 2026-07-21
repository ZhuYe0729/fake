#!/usr/bin/env python3
"""Fit a non-negative quantization/sparsity/interation real-vLLM NLL proxy."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from common import DEBUG, ERRORS, PARTS, SOURCE, TYPES


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [a - p for a, p in zip(actual, predicted)]
    def rank(values: list[float]) -> list[float]:
        return [1 + sum(other < value for other in values) + .5 * sum(other == value for other in values) for value in values]
    ra, rp = rank(actual), rank(predicted); ma, mp = sum(ra) / len(ra), sum(rp) / len(rp)
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((x - mp) ** 2 for x in rp))
    return {"mae": sum(abs(error) for error in errors) / len(errors), "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)), "signed_error": sum(errors) / len(errors), "spearman": 0.0 if not den else sum((x - ma) * (y - mp) for x, y in zip(ra, rp)) / den}


def errors(method: str) -> dict[tuple[int, str], float]:
    rows = read_csv(ERRORS); out = {}
    for layer in range(32):
        for typ, parts in PARTS.items():
            values = [float(row["local_rel_mse"]) for row in rows if int(row["layer"]) == layer and row["module_type"] in parts and row["method"] == method]
            out[layer, typ] = sum(values) / len(values)
    return out


def signals(policy: dict, quant: dict[tuple[int, str], float], sparse: dict[tuple[int, str], float]) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.zeros((4, 4), dtype=torch.float64); s = torch.zeros((4, 4), dtype=torch.float64)
    for name, entry in policy["method_map"].items():
        layer, typ, method = int(name.split(".")[2]), name.rsplit(".", 1)[-1], entry["prefill_method"]
        bucket, type_index = layer // 8, TYPES.index(typ)
        if method in {"dense_nvfp4", "sparse_nvfp4"}:
            q[bucket, type_index] += quant[layer, typ]
        if method in {"sparse_bf16", "sparse_nvfp4"}:
            s[bucket, type_index] += sparse[layer, typ]
    return q, s


def main() -> None:
    old_manifest = json.loads((SOURCE / "policies/prefill_only/manifest.json").read_text())
    old_labels = {row["policy_id"]: row for row in read_csv(SOURCE / "nll/prefill_only.csv")}
    new_manifest = json.loads((DEBUG / "manifest.json").read_text())
    new_labels = {row["policy_id"]: row for row in read_csv(DEBUG / "nll.csv")}
    q_error, s_error = errors("dense_nvfp4"), errors("sparse_bf16")
    entries = []
    for item in old_manifest:
        entries.append({"id": item["policy_id"], "group": "old_train" if item["split"] == "train" else "old_holdout", "policy": json.loads(Path(item["path"]).read_text()), "y": float(old_labels[item["policy_id"]]["target_delta_nll"])})
    for item in new_manifest:
        entries.append({"id": item["policy_id"], "group": "mechanism_train" if item["split"] == "train" else "mechanism_holdout", "policy": json.loads(Path(item["path"]).read_text()), "y": float(new_labels[item["policy_id"]]["delta_nll"])})
    q = torch.stack([signals(entry["policy"], q_error, s_error)[0] for entry in entries])
    s = torch.stack([signals(entry["policy"], q_error, s_error)[1] for entry in entries])
    y = torch.tensor([entry["y"] for entry in entries], dtype=torch.float64)
    train = torch.tensor([entry["group"] in {"old_train", "mechanism_train"} for entry in entries])
    scale = (q[train].sum((1, 2)) + s[train].sum((1, 2))).mean().clamp(min=1e-12)
    q, s = q / scale, s / scale
    # ReLU gives the required non-negative coefficients while retaining a true
    # zero-error anchor.  softplus(0) is positive and incorrectly penalizes a
    # policy made only of calibrated low-sensitivity modules.
    q_weight = torch.full((4, 4), .01, dtype=torch.float64, requires_grad=True)
    s_weight = torch.full((4, 4), .01, dtype=torch.float64, requires_grad=True)
    sparse_accumulation = torch.full((4,), .01, dtype=torch.float64, requires_grad=True)
    interaction = torch.full((4,), .01, dtype=torch.float64, requires_grad=True)
    parameters = [q_weight, s_weight, sparse_accumulation, interaction]
    optimizer = torch.optim.Adam(parameters, lr=.025)
    for _ in range(5000):
        optimizer.zero_grad()
        qw, sw = torch.relu(q_weight), torch.relu(s_weight)
        a, c = torch.relu(sparse_accumulation), torch.relu(interaction)
        qg, sg = q.sum(2), s.sum(2)
        prediction = (q * qw).sum((1, 2)) + (s * sw).sum((1, 2)) + (sg.square() * a).sum(1) + (sg * qg * c).sum(1)
        loss = ((prediction[train] - y[train]).square()).mean() + .0001 * sum(parameter.square().mean() for parameter in parameters)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        qw, sw = torch.relu(q_weight), torch.relu(s_weight)
        a, c = torch.relu(sparse_accumulation), torch.relu(interaction)
        qg, sg = q.sum(2), s.sum(2)
        prediction = (q * qw).sum((1, 2)) + (s * sw).sum((1, 2)) + (sg.square() * a).sum(1) + (sg * qg * c).sum(1)
    old_v1 = {row["policy_id"]: float(row["predicted_delta_nll"]) for row in read_csv(SOURCE / "reports/quality/predictions.csv")}
    rows = []
    for index, entry in enumerate(entries):
        row = {"policy_id": entry["id"], "group": entry["group"], "actual_delta_nll": entry["y"], "v2_predicted_delta_nll": float(prediction[index]), "v2_residual": entry["y"] - float(prediction[index]), "v1_predicted_delta_nll": old_v1.get(entry["id"], "")}
        rows.append(row)
    report = DEBUG / "report"; report.mkdir(parents=True, exist_ok=True)
    with (report / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped = {group: metric([row["actual_delta_nll"] for row in rows if row["group"] == group], [row["v2_predicted_delta_nll"] for row in rows if row["group"] == group]) for group in sorted({row["group"] for row in rows})}
    old_holdout = [row for row in rows if row["group"] == "old_holdout"]
    comparison = {"v1_old_holdout": metric([row["actual_delta_nll"] for row in old_holdout], [float(row["v1_predicted_delta_nll"]) for row in old_holdout]), "v2_old_holdout": metric([row["actual_delta_nll"] for row in old_holdout], [row["v2_predicted_delta_nll"] for row in old_holdout])}
    model = {"proxy": "nonnegative per-bucket/type quant + sparse + sparse-squared + sparse-quant interaction", "parameterization": "ReLU nonnegative weights with exact zero anchor", "feature_scale": float(scale), "quant_weight": qw.tolist(), "sparse_weight": sw.tolist(), "sparse_accumulation": a.tolist(), "interaction": c.tolist(), "fit_groups": ["old_train", "mechanism_train"], "zero_bias": True}
    (report / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (report / "metrics.json").write_text(json.dumps({"v2": grouped, "old_holdout_comparison": comparison}, indent=2) + "\n")
    print(json.dumps({"v2": grouped, "old_holdout_comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
