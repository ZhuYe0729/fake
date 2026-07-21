#!/usr/bin/env python3
"""Fit the unchanged positive local+global proxy to real-vLLM NLL labels."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from common import METHODS, MODELS, TYPES, model_root


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ranks(values: list[float]) -> list[float]:
    return [1 + sum(other < value for other in values) + 0.5 * sum(other == value for other in values) for value in values]


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [a - p for a, p in zip(actual, predicted)]
    ra, rp = ranks(actual), ranks(predicted)
    ma, mp = sum(ra) / len(ra), sum(rp) / len(rp)
    denominator = math.sqrt(sum((item - ma) ** 2 for item in ra) * sum((item - mp) ** 2 for item in rp))
    return {"mae": sum(abs(error) for error in errors) / len(errors), "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)), "mean_signed_error": sum(errors) / len(errors), "spearman": sum((x - ma) * (y - mp) for x, y in zip(ra, rp)) / denominator if denominator else 0.0}


def llama2_errors() -> dict[tuple[int, str, str], float]:
    rows = read_csv(MODELS["llama2"]["local_error_source"])
    parts = {"qkv_proj": {"q_proj", "k_proj", "v_proj"}, "o_proj": {"o_proj"}, "gate_up_proj": {"gate_proj", "up_proj"}, "down_proj": {"down_proj"}}
    aliases = {"w4a16_ours": "dense_nvfp4"}
    out = {}
    for layer in range(32):
        for typ, fused_parts in parts.items():
            for method in METHODS:
                if method == "dense_bf16":
                    out[layer // 8, typ, method] = 0.0
                else:
                    values = [float(row["local_rel_mse"]) for row in rows if int(row["layer"]) == layer and row["module_type"] in fused_parts and row["method"] == aliases.get(method, method)]
                    # Existing Llama2 feature aggregation first averages a
                    # fused type within layer, then sums its layers in a bucket.
                    out[layer // 8, typ, method] = out.get((layer // 8, typ, method), 0.0) + sum(values) / len(values)
    return out


def llama31_errors() -> dict[tuple[int, str, str], float]:
    source = MODELS["llama31"]["local_error_source"]
    out = {}
    for method in METHODS:
        if method == "dense_bf16":
            for bucket in range(4):
                for typ in TYPES:
                    out[bucket, typ, method] = 0.0
            continue
        for row in read_csv(source / f"prefill_{method}.csv"):
            out[int(row["layer_bucket"]), row["fused_type"], method] = float(row["output_rel_mse"])
    return out


def features(policy: dict, errors: dict[tuple[int, str, str], float], model: str) -> torch.Tensor:
    tensor = torch.zeros((len(METHODS), 4, len(TYPES)), dtype=torch.float64)
    # Llama2's historic source already aggregates its local error over layers
    # in a bucket; Llama3's source stores a bucket aggregate as well.
    seen: set[tuple[int, str]] = set()
    for name, entry in policy["method_map"].items():
        layer, typ = int(name.split(".")[2]), name.rsplit(".", 1)[-1]
        bucket, method = layer // 8, entry["prefill_method"]
        key = (bucket, typ)
        if model == "llama2":
            if key in seen:
                continue
            seen.add(key)
        tensor[METHODS.index(method), bucket, TYPES.index(typ)] += errors[bucket, typ, method]
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    root = model_root(args.model)
    manifest = json.loads((root / "policies/prefill_only/manifest.json").read_text())
    labels = {row["policy_id"]: row for row in read_csv(root / "nll/prefill_only.csv")}
    if set(labels) != {row["policy_id"] for row in manifest}:
        raise RuntimeError("labels do not cover the frozen manifest")
    errors = llama2_errors() if args.model == "llama2" else llama31_errors()
    policies = [json.loads(Path(row["path"]).read_text()) for row in manifest]
    X = torch.stack([features(policy, errors, args.model) for policy in policies])
    y = torch.tensor([float(labels[row["policy_id"]]["target_delta_nll"]) for row in manifest], dtype=torch.float64)
    train = torch.tensor([row["split"] == "train" for row in manifest])
    scale = X[train].sum((1, 2, 3)).mean().clamp(min=1e-12)
    X = X / scale
    global_factor = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    method_factor = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    bucket_factor = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    type_factor = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    parameters = [global_factor, method_factor, bucket_factor, type_factor, bias]
    optimizer = torch.optim.Adam(parameters, lr=0.03)
    for _ in range(args.steps):
        optimizer.zero_grad()
        coefficients = torch.nn.functional.softplus(global_factor + method_factor[:, None, None] + bucket_factor[None, :, None] + type_factor[None, None, :])
        prediction = bias + (X * coefficients).sum((1, 2, 3))
        loss = ((prediction[train] - y[train]) ** 2).mean() + 0.05 * sum((parameter * parameter).mean() for parameter in parameters[:-1])
        loss.backward(); optimizer.step()
    with torch.no_grad():
        coefficients = torch.nn.functional.softplus(global_factor + method_factor[:, None, None] + bucket_factor[None, :, None] + type_factor[None, None, :])
        prediction = (bias + (X * coefficients).sum((1, 2, 3))).tolist()
    rows = []
    for index, item in enumerate(manifest):
        policy = policies[index]
        counts = {method: sum(entry["prefill_method"] == method for entry in policy["method_map"].values()) for method in METHODS}
        rows.append({"policy_id": item["policy_id"], "split": item["split"], "policy_kind": item["policy_kind"], "actual_delta_nll": float(y[index]), "predicted_delta_nll": prediction[index], "residual_actual_minus_predicted": float(y[index]) - prediction[index], **{f"count_{method}": counts[method] for method in METHODS}})
    report = root / "reports/quality"
    report.mkdir(parents=True, exist_ok=True)
    with (report / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {split: metrics([row["actual_delta_nll"] for row in rows if row["split"] == split], [row["predicted_delta_nll"] for row in rows if row["split"] == split]) for split in ("train", "holdout")}
    by_kind = {kind: metrics([row["actual_delta_nll"] for row in rows if row["policy_kind"] == kind], [row["predicted_delta_nll"] for row in rows if row["policy_kind"] == kind]) for kind in sorted({row["policy_kind"] for row in rows})}
    result = {"model": args.model, "label_backend": "real vLLM direct prompt-logprob", "feature_backend": "existing local-error table", "proxy": "positive normalized local-error aggregation + global/method/bucket/type factors", "fit_split": "p00-p53", "sample_blocks": int(labels["p00"]["sample_count"]), "metrics": summary, "metrics_by_policy_kind": by_kind}
    (report / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    model = {"feature_scale": float(scale), "global": float(global_factor.detach()), "method": method_factor.detach().tolist(), "bucket": bucket_factor.detach().tolist(), "type": type_factor.detach().tolist(), "bias": float(bias.detach()), "coefficients": coefficients.detach().tolist(), "fit_split": "p00-p53 only"}
    (report / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
