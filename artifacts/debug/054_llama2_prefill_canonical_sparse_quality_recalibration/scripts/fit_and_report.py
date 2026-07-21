#!/usr/bin/env python3
"""Fit the unchanged 046 proxy using canonical sparse NLL labels/features."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"))
LEGACY_FIT = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/fit_quality_proxy.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [a - p for a, p in zip(actual, predicted)]
    ranks = lambda values: [1 + sum(other < value for other in values) + .5 * sum(other == value for other in values) for value in values]
    ra, rp = ranks(actual), ranks(predicted)
    ma, mp = sum(ra) / len(ra), sum(rp) / len(rp)
    denominator = math.sqrt(sum((item - ma) ** 2 for item in ra) * sum((item - mp) ** 2 for item in rp))
    return {"mae": sum(abs(item) for item in errors) / len(errors),
            "rmse": math.sqrt(sum(item * item for item in errors) / len(errors)),
            "mean_signed_error": sum(errors) / len(errors),
            "spearman": sum((x - ma) * (y - mp) for x, y in zip(ra, rp)) / denominator if denominator else 0.0}


def load_feature_module():
    sys.path.insert(0, str(LEGACY_FIT.parent))
    spec = importlib.util.spec_from_file_location("fit_046", LEGACY_FIT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import 046 proxy")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def canonical_errors(feature) -> dict[tuple[int, str, str], float]:
    rows = read_csv(EXPERIMENT / "local_errors/module_method_errors.csv")
    parts = {"qkv_proj": {"q_proj", "k_proj", "v_proj"}, "o_proj": {"o_proj"},
             "gate_up_proj": {"gate_proj", "up_proj"}, "down_proj": {"down_proj"}}
    aliases = {"w4a16_ours": "dense_nvfp4"}
    result = {}
    for layer in range(32):
        for typ, part_set in parts.items():
            for method in feature.METHODS:
                if method == "dense_bf16":
                    result[layer // 8, typ, method] = 0.0
                    continue
                values = [float(row["local_rel_mse"]) for row in rows
                          if int(row["layer"]) == layer and row["module_type"] in part_set
                          and row["method"] == aliases.get(method, method)]
                if not values:
                    raise RuntimeError(f"missing local feature: layer={layer} type={typ} method={method}")
                key = (layer // 8, typ, method)
                result[key] = result.get(key, 0.0) + sum(values) / len(values)
    return result


def main() -> None:
    feature = load_feature_module()
    manifest = json.loads((EXPERIMENT / "policies/prefill_only/manifest.json").read_text())
    labels = {row["policy_id"]: row for row in read_csv(EXPERIMENT / "nll/prefill_only.csv")}
    policies = [json.loads(Path(row["path"]).read_text()) for row in manifest]
    errors = canonical_errors(feature)
    X = torch.stack([feature.features(policy, errors, "llama2") for policy in policies])
    y = torch.tensor([float(labels[row["policy_id"]]["target_delta_nll"]) for row in manifest], dtype=torch.float64)
    train = torch.tensor([row["split"] == "train" for row in manifest])
    scale = X[train].sum((1, 2, 3)).mean().clamp(min=1e-12); X = X / scale
    global_factor = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    method_factor = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    bucket_factor = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    type_factor = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    parameters = [global_factor, method_factor, bucket_factor, type_factor, bias]
    optimizer = torch.optim.Adam(parameters, lr=.03)
    for _ in range(3000):
        optimizer.zero_grad()
        coefficients = torch.nn.functional.softplus(global_factor + method_factor[:, None, None] + bucket_factor[None, :, None] + type_factor[None, None, :])
        prediction = bias + (X * coefficients).sum((1, 2, 3))
        loss = ((prediction[train] - y[train]) ** 2).mean() + .05 * sum((item * item).mean() for item in parameters[:-1])
        loss.backward(); optimizer.step()
    with torch.no_grad():
        coefficients = torch.nn.functional.softplus(global_factor + method_factor[:, None, None] + bucket_factor[None, :, None] + type_factor[None, None, :])
        prediction = (bias + (X * coefficients).sum((1, 2, 3))).tolist()
    rows = []
    for index, item in enumerate(manifest):
        policy = policies[index]
        counts = {method: sum(value["prefill_method"] == method for value in policy["method_map"].values()) for method in feature.METHODS}
        rows.append({"policy_id": item["policy_id"], "split": item["split"], "policy_kind": item["policy_kind"],
                     "actual_delta_nll": float(y[index]), "predicted_delta_nll": prediction[index],
                     "residual_actual_minus_predicted": float(y[index]) - prediction[index],
                     **{f"count_{method}": counts[method] for method in feature.METHODS}})
    report = EXPERIMENT / "reports/quality"; report.mkdir(parents=True, exist_ok=True)
    with (report / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"proxy": "unchanged positive normalized local-error aggregation + global/method/bucket/type softplus factors",
              "label_backend": "canonical sparse phase_hetero real-vLLM NLL", "feature_backend": "canonical sparse wrapper local errors + legacy non-sparse rows",
              "fit_split": "p00-p53", "metrics": {split: metrics([row["actual_delta_nll"] for row in rows if row["split"] == split], [row["predicted_delta_nll"] for row in rows if row["split"] == split]) for split in ("train", "holdout")}}
    (report / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (report / "model.json").write_text(json.dumps({"feature_scale": float(scale), "global": float(global_factor.detach()), "method": method_factor.detach().tolist(), "bucket": bucket_factor.detach().tolist(), "type": type_factor.detach().tolist(), "bias": float(bias.detach()), "coefficients": coefficients.detach().tolist(), "fit_split": "p00-p53"}, indent=2) + "\n")
    for split, color, marker in (("train", "#4c78a8", "o"), ("holdout", "#e45756", "s")):
        subset = [row for row in rows if row["split"] == split]
        plt.scatter([row["actual_delta_nll"] for row in subset], [row["predicted_delta_nll"] for row in subset], color=color, marker=marker, label=split)
    upper = max(max(row[key] for row in rows) for key in ("actual_delta_nll", "predicted_delta_nll"))
    plt.plot([0, upper], [0, upper], "--", color="#555555"); plt.xlabel("Measured canonical-sparse ΔNLL"); plt.ylabel("Predicted ΔNLL"); plt.legend(); plt.tight_layout(); plt.savefig(report / "predicted_vs_measured.png", dpi=180); plt.close()
    lines = ["# Llama2 canonical sparse quality recalibration", "", "| split | MAE | RMSE | Spearman |", "|---|---:|---:|---:|"]
    lines += [f"| {split} | {result['metrics'][split]['mae']:.6f} | {result['metrics'][split]['rmse']:.6f} | {result['metrics'][split]['spearman']:.4f} |" for split in ("train", "holdout")]
    lines += ["", "## Uniform controls", "", "| policy | measured ΔNLL | predicted ΔNLL | residual |", "|---|---:|---:|---:|"]
    lines += [f"| {row['policy_id']} | {row['actual_delta_nll']:.6f} | {row['predicted_delta_nll']:.6f} | {row['residual_actual_minus_predicted']:.6f} |" for row in rows[:5]]
    (report / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
