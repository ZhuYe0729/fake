#!/usr/bin/env python3
"""Fit the prefill-only ReLU local+global quality proxy from canonical labels."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from scenario import EXP, LOCAL_ERRORS, METHODS, TYPES

def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))

def metric(actual: list[float], predicted: list[float]) -> dict[str, float]:
    residual = [a - p for a, p in zip(actual, predicted)]
    rank = lambda xs: [1 + sum(y < x for y in xs) + .5 * sum(y == x for y in xs) for x in xs]
    ra, rp = rank(actual), rank(predicted); ma, mp = sum(ra) / len(ra), sum(rp) / len(rp)
    denom = math.sqrt(sum((x-ma)**2 for x in ra) * sum((x-mp)**2 for x in rp))
    return {"mae": sum(abs(x) for x in residual) / len(residual), "rmse": math.sqrt(sum(x*x for x in residual) / len(residual)),
            "mean_signed_error": sum(residual) / len(residual), "spearman": sum((x-ma)*(y-mp) for x, y in zip(ra, rp)) / denom if denom else 0.0}

def local_errors() -> dict[tuple[int, str, str], float]:
    result = {(bucket, typ, "dense_bf16"): 0.0 for bucket in range(4) for typ in TYPES}
    for method in METHODS[1:]:
        for row in rows(LOCAL_ERRORS / f"prefill_{method}.csv"):
            result[int(row["layer_bucket"]), row["fused_type"], method] = float(row["output_rel_mse"])
    if len(result) != len(METHODS) * 4 * len(TYPES): raise RuntimeError("incomplete local feature table")
    return result

def features(policy: dict, errors: dict[tuple[int, str, str], float]) -> torch.Tensor:
    value = torch.zeros((len(METHODS), 4, len(TYPES)), dtype=torch.float64)
    for name, item in policy["method_map"].items():
        bucket = int(name.split(".")[2]) // 8; raw = name.rsplit(".", 1)[-1]
        typ = "qkv_proj" if raw in {"q_proj", "k_proj", "v_proj", "qkv_proj"} else "gate_up_proj" if raw in {"gate_proj", "up_proj", "gate_up_proj"} else raw
        method = item["prefill_method"]
        value[METHODS.index(method), bucket, TYPES.index(typ)] += errors[bucket, typ, method]
    return value

def main() -> None:
    manifest = json.loads((EXP / "policies/prefill_only/manifest.json").read_text())
    labels = {row["policy_id"]: row for row in rows(EXP / "nll/prefill_only.csv")}
    policies = [json.loads(Path(row["path"]).read_text()) for row in manifest]
    errors = local_errors()
    x = torch.stack([features(policy, errors) for policy in policies])
    y = torch.tensor([float(labels[row["policy_id"]]["target_delta_nll"]) for row in manifest], dtype=torch.float64)
    train = torch.tensor([row["split"] == "train" for row in manifest]); scale = x[train].sum((1,2,3)).mean().clamp(min=1e-12); x = x / scale
    global_factor = torch.tensor([.1], dtype=torch.float64, requires_grad=True)
    method = torch.zeros(len(METHODS), dtype=torch.float64, requires_grad=True); bucket = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    typ = torch.zeros(len(TYPES), dtype=torch.float64, requires_grad=True); bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    parameters = [global_factor, method, bucket, typ, bias]; optim = torch.optim.Adam(parameters, lr=.03)
    for _ in range(4000):
        optim.zero_grad(); coef = torch.relu(global_factor + method[:,None,None] + bucket[None,:,None] + typ[None,None,:])
        prediction = bias + (x * coef).sum((1,2,3)); loss = ((prediction[train]-y[train])**2).mean() + .05 * sum((p*p).mean() for p in parameters[:-1]); loss.backward(); optim.step()
    with torch.no_grad():
        coef = torch.relu(global_factor + method[:,None,None] + bucket[None,:,None] + typ[None,None,:]); prediction = (bias + (x*coef).sum((1,2,3))).tolist()
    report = EXP / "reports/quality"; report.mkdir(parents=True, exist_ok=True); output = []
    for index, row in enumerate(manifest):
        actual = float(y[index]); output.append({"policy_id": row["policy_id"], "split": row["split"], "policy_kind": row["policy_kind"], "actual_delta_nll": actual, "predicted_delta_nll": prediction[index], "residual_actual_minus_predicted": actual-prediction[index]})
    with (report / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    result = {"proxy": "prefill-only positive local-error aggregation: global + method + layer-bucket + fused-type ReLU calibration", "label_backend": "canonical sparse phase_hetero real-vLLM full-prefill NLL", "feature_backend": "057 canonical prefill local output_rel_mse", "fit_split": "p00-p53", "metrics": {split: metric([r["actual_delta_nll"] for r in output if r["split"]==split], [r["predicted_delta_nll"] for r in output if r["split"]==split]) for split in ("train", "holdout")}}
    (report / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    model = {"feature_scale": float(scale), "global": float(global_factor.detach()), "method": method.detach().tolist(), "bucket": bucket.detach().tolist(), "type": typ.detach().tolist(), "bias": float(bias.detach()), "coefficients": coef.detach().tolist()}
    (report / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    for split, color, marker in (("train", "#4c78a8", "o"), ("holdout", "#e45756", "s")):
        subset = [r for r in output if r["split"] == split]; plt.scatter([r["actual_delta_nll"] for r in subset], [r["predicted_delta_nll"] for r in subset], color=color, marker=marker, label=split)
    upper = max(max(r[k] for r in output) for k in ("actual_delta_nll", "predicted_delta_nll")); plt.plot([0,upper],[0,upper],"--",color="#555"); plt.xlabel("Measured real-vLLM ΔNLL"); plt.ylabel("Predicted ΔNLL"); plt.legend(); plt.tight_layout(); plt.savefig(report / "predicted_vs_measured.png", dpi=180); plt.close()
    print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
