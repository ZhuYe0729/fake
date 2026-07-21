#!/usr/bin/env python3
"""Fit the unchanged 046 proxy and compare it with the legacy-label model."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "artifacts/debug/053_llama2_prefill_phase_unified_quality_recalibration/llama2_7b_chat"
LEGACY = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat"
LEGACY_FIT = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/fit_quality_proxy.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [a - p for a, p in zip(actual, predicted)]
    ranks = lambda values: [1 + sum(other < value for other in values) + .5 * sum(other == value for other in values) for value in values]
    ra, rp = ranks(actual), ranks(predicted)
    ma, mp = sum(ra) / len(ra), sum(rp) / len(rp)
    denominator = math.sqrt(sum((value - ma) ** 2 for value in ra) * sum((value - mp) ** 2 for value in rp))
    return {"mae": sum(abs(error) for error in errors) / len(errors), "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)), "mean_signed_error": sum(errors) / len(errors), "spearman": sum((x - ma) * (y - mp) for x, y in zip(ra, rp)) / denominator if denominator else 0.0}


def load_feature_module():
    scripts = str(LEGACY_FIT.parent)
    import sys
    sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("legacy_fit_046", LEGACY_FIT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load 046 feature module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    feature = load_feature_module()
    manifest = json.loads((EXPERIMENT / "policies/prefill_only/manifest.json").read_text())
    labels = {row["policy_id"]: row for row in read_csv(EXPERIMENT / "nll/prefill_only.csv")}
    policies = [json.loads(Path(row["path"]).read_text()) for row in manifest]
    errors = feature.llama2_errors()
    X = torch.stack([feature.features(policy, errors, "llama2") for policy in policies])
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
    optimizer = torch.optim.Adam(parameters, lr=.03)
    for _ in range(3000):
        optimizer.zero_grad()
        coefficient = torch.nn.functional.softplus(global_factor + method_factor[:, None, None] + bucket_factor[None, :, None] + type_factor[None, None, :])
        prediction = bias + (X * coefficient).sum((1, 2, 3))
        loss = ((prediction[train] - y[train]) ** 2).mean() + .05 * sum((parameter * parameter).mean() for parameter in parameters[:-1])
        loss.backward(); optimizer.step()
    with torch.no_grad():
        coefficient = torch.nn.functional.softplus(global_factor + method_factor[:, None, None] + bucket_factor[None, :, None] + type_factor[None, None, :])
        prediction = (bias + (X * coefficient).sum((1, 2, 3))).tolist()
    rows = []
    for index, item in enumerate(manifest):
        policy = policies[index]
        counts = {method: sum(value["prefill_method"] == method for value in policy["method_map"].values()) for method in feature.METHODS}
        rows.append({"policy_id": item["policy_id"], "split": item["split"], "policy_kind": item["policy_kind"], "actual_delta_nll": float(y[index]), "predicted_delta_nll": prediction[index], "residual_actual_minus_predicted": float(y[index]) - prediction[index], **{f"count_{method}": counts[method] for method in feature.METHODS}})
    report = EXPERIMENT / "reports/quality"; report.mkdir(parents=True, exist_ok=True)
    with (report / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"model": "llama2", "label_backend": "phase_hetero_mytest for all compressed policies; raw BF16 p00 reference", "feature_backend": "unchanged 046 local-error table", "proxy": "unchanged positive normalized local-error aggregation + global/method/bucket/type factors", "fit_split": "p00-p53", "metrics": {split: metrics([row["actual_delta_nll"] for row in rows if row["split"] == split], [row["predicted_delta_nll"] for row in rows if row["split"] == split]) for split in ("train", "holdout")}, "metrics_by_policy_kind": {kind: metrics([row["actual_delta_nll"] for row in rows if row["policy_kind"] == kind], [row["predicted_delta_nll"] for row in rows if row["policy_kind"] == kind]) for kind in sorted({row["policy_kind"] for row in rows})}}
    legacy = json.loads((LEGACY / "reports/quality/metrics.json").read_text())
    legacy_labels = {row["policy_id"]: row for row in read_csv(LEGACY / "nll/prefill_only.csv")}
    label_rows = [{"policy_id": row["policy_id"], "policy_kind": row["policy_kind"], "legacy_delta_nll": float(legacy_labels[row["policy_id"]]["target_delta_nll"]), "phase_unified_delta_nll": row["actual_delta_nll"], "phase_minus_legacy": row["actual_delta_nll"] - float(legacy_labels[row["policy_id"]]["target_delta_nll"])} for row in rows]
    with (report / "label_comparison_046_vs_053.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0])); writer.writeheader(); writer.writerows(label_rows)
    result["legacy_046_metrics"] = legacy["metrics"]
    result["legacy_046_metrics_by_policy_kind"] = legacy["metrics_by_policy_kind"]
    (report / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (report / "model.json").write_text(json.dumps({"feature_scale": float(scale), "global": float(global_factor.detach()), "method": method_factor.detach().tolist(), "bucket": bucket_factor.detach().tolist(), "type": type_factor.detach().tolist(), "bias": float(bias.detach()), "coefficients": coefficient.detach().tolist(), "fit_split": "p00-p53 only"}, indent=2) + "\n")
    for split, color, marker in (("train", "#4c78a8", "o"), ("holdout", "#e45756", "s")):
        subset = [row for row in rows if row["split"] == split]
        plt.scatter([row["actual_delta_nll"] for row in subset], [row["predicted_delta_nll"] for row in subset], color=color, marker=marker, label=split)
    upper = max(max(row[key] for row in rows) for key in ("actual_delta_nll", "predicted_delta_nll"))
    plt.plot([0, upper], [0, upper], "--", color="#555555"); plt.xlabel("Measured phase-unified ΔNLL"); plt.ylabel("Predicted ΔNLL"); plt.legend(); plt.tight_layout(); plt.savefig(report / "predicted_vs_measured.png", dpi=180); plt.close()
    lines = ["# Llama2 phase-unified quality recalibration", "", "The proxy formula and 54/18 split are unchanged from 046; compressed labels are rebuilt with phase runtime.", "", "| split | 046 MAE | 053 MAE | 046 RMSE | 053 RMSE | 046 Spearman | 053 Spearman |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "holdout"):
        old, new = legacy["metrics"][split], result["metrics"][split]
        lines.append(f"| {split} | {old['mae']:.6f} | {new['mae']:.6f} | {old['rmse']:.6f} | {new['rmse']:.6f} | {old['spearman']:.4f} | {new['spearman']:.4f} |")
    lines += ["", "## Labels changed by the corrected pipeline", "", "| policy | legacy ΔNLL | phase-unified ΔNLL | difference |", "|---|---:|---:|---:|"]
    for row in label_rows:
        if abs(row["phase_minus_legacy"]) > 1e-9:
            lines.append(f"| {row['policy_id']} | {row['legacy_delta_nll']:.6f} | {row['phase_unified_delta_nll']:.6f} | {row['phase_minus_legacy']:+.6f} |")
    lines += ["", "## Uniform controls", "", "| policy | measured ΔNLL | predicted ΔNLL | residual |", "|---|---:|---:|---:|"]
    for row in rows[:5]: lines.append(f"| {row['policy_id']} | {row['actual_delta_nll']:.6f} | {row['predicted_delta_nll']:.6f} | {row['residual_actual_minus_predicted']:.6f} |")
    (report / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
