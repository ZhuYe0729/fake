#!/usr/bin/env python3
"""Original 046 proxy with only softplus-to-ReLU coefficient change."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration"
sys.path.insert(0, str(SOURCE / "scripts"))
from common import METHODS, MODELS, TYPES, model_root  # noqa: E402
from fit_quality_proxy import llama2_errors, metrics, read_csv, features  # noqa: E402


def main() -> None:
    root = model_root("llama2")
    manifest = json.loads((root / "policies/prefill_only/manifest.json").read_text())
    labels = {row["policy_id"]: row for row in read_csv(root / "nll/prefill_only.csv")}
    errors = llama2_errors(); policies = [json.loads(Path(item["path"]).read_text()) for item in manifest]
    x = torch.stack([features(policy, errors, "llama2") for policy in policies]); y = torch.tensor([float(labels[item["policy_id"]]["target_delta_nll"]) for item in manifest], dtype=torch.float64)
    train = torch.tensor([item["split"] == "train" for item in manifest]); scale = x[train].sum((1, 2, 3)).mean().clamp(min=1e-12); x = x / scale
    # Same factorization, bias, optimizer, steps, and L2 as 046.  The small
    # positive start is solely needed because ReLU has zero derivative at 0.
    global_factor = torch.full((1,), .01, dtype=torch.float64, requires_grad=True)
    method_factor = torch.full((5,), .01, dtype=torch.float64, requires_grad=True)
    bucket_factor = torch.full((4,), .01, dtype=torch.float64, requires_grad=True)
    type_factor = torch.full((4,), .01, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    parameters = [global_factor, method_factor, bucket_factor, type_factor, bias]; optimizer = torch.optim.Adam(parameters, lr=.03)
    for _ in range(3000):
        optimizer.zero_grad(); coefficients = torch.relu(global_factor + method_factor[:, None, None] + bucket_factor[None, :, None] + type_factor[None, None, :]); prediction = bias + (x * coefficients).sum((1, 2, 3))
        (((prediction[train] - y[train]).square()).mean() + .05 * sum((value * value).mean() for value in parameters[:-1])).backward(); optimizer.step()
    with torch.no_grad():
        coefficients = torch.relu(global_factor + method_factor[:, None, None] + bucket_factor[None, :, None] + type_factor[None, None, :]); prediction = (bias + (x * coefficients).sum((1, 2, 3))).tolist()
    rows = [{"policy_id": item["policy_id"], "split": item["split"], "actual_delta_nll": float(y[index]), "predicted_delta_nll": prediction[index], "residual": float(y[index]) - prediction[index]} for index, item in enumerate(manifest)]
    output = ROOT / "artifacts/debug/050_llama2_original_relu_quality_proxy/report"; output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"model": "046 original proxy; coefficient activation softplus -> ReLU only", "metrics": {split: metrics([row["actual_delta_nll"] for row in rows if row["split"] == split], [row["predicted_delta_nll"] for row in rows if row["split"] == split]) for split in ("train", "holdout")}, "feature_scale": float(scale), "bias": float(bias.detach())}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
