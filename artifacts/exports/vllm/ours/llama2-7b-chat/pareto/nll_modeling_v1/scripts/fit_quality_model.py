#!/usr/bin/env python3
"""Fit and report the v1 positive additive NLL proxy on fixed policy holdouts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=("prefill_only", "prefill_decode"), required=True)
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--lr", type=float, default=.03); p.add_argument("--l2", type=float, default=.02)
    return p.parse_args()


def csv_rows(path: Path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f: w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def fused_errors(root: Path) -> dict[tuple[int, str, str], float]:
    source = root.parents[5] / "debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv"
    raw = csv_rows(source); result = {}
    method_map = {"w4a16_ours": "dense_nvfp4"}
    for layer in range(32):
        for typ, parts in {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "o_proj": ("o_proj",), "gate_up_proj": ("gate_proj", "up_proj"), "down_proj": ("down_proj",)}.items():
            for method in METHODS:
                if method == "dense_bf16": result[layer, typ, method] = 0.; continue
                selected = [r for r in raw if int(r["layer"]) == layer and r["module_type"] in parts and r["method"] == method_map.get(method, method)]
                result[layer, typ, method] = sum(float(r["local_rel_mse"]) * float(r["numel"]) for r in selected)
    return result


def design(policy: dict, phase: str, errors: dict) -> torch.Tensor:
    x = torch.zeros((len(METHODS), 32, len(TYPES)), dtype=torch.float64)
    for name, entry in policy["method_map"].items():
        layer = int(name.split(".")[2]); typ = name.rsplit(".", 1)[-1]; method = entry[f"{phase}_method"]
        x[METHODS.index(method), layer, TYPES.index(typ)] += errors[layer, typ, method]
    return x


def ranks(values: list[float]) -> list[float]:
    return [1 + sum(v < x for v in values) + .5 * sum(v == x for v in values) for x in values]
def metric(actual: list[float], predicted: list[float]) -> dict:
    n = max(len(actual), 1); err = [p - y for p, y in zip(predicted, actual)]
    ra, rp = ranks(actual), ranks(predicted); ma, mp = sum(ra)/n, sum(rp)/n
    denom = math.sqrt(sum((x-ma)**2 for x in ra) * sum((x-mp)**2 for x in rp))
    return {"count": n, "mae": sum(abs(x) for x in err)/n, "rmse": math.sqrt(sum(x*x for x in err)/n), "spearman": sum((x-ma)*(y-mp) for x,y in zip(ra,rp))/denom if denom else 0.}


def main() -> None:
    a = parse(); root = a.output_root; errors = fused_errors(root)
    manifest = json.loads((root / "policies" / a.scenario / "manifest.json").read_text()); policy_map = {x["policy_id"]: json.loads(Path(x["path"]).read_text()) for x in manifest}
    nll = {x["policy_id"]: x for x in csv_rows(root / "nll" / f"{a.scenario}.csv")}
    missing = [x["policy_id"] for x in manifest if x["policy_id"] not in nll]
    if missing: raise RuntimeError(f"missing NLL policies: {missing}")
    phase_weights = [("prefill", 1.)] if a.scenario == "prefill_only" else [("prefill", 1.), ("decode", 80.)]
    xs = torch.stack([sum((weight * design(policy_map[x["policy_id"]], phase, errors) for phase, weight in phase_weights), torch.zeros((5,32,4), dtype=torch.float64)) for x in manifest])
    y = torch.tensor([float(nll[x["policy_id"]]["target_delta_nll"]) for x in manifest], dtype=torch.float64)
    train = torch.tensor([x["split"] == "train" for x in manifest]);
    log_global = torch.zeros(1, dtype=torch.float64, requires_grad=True); log_method = torch.zeros(5, dtype=torch.float64, requires_grad=True); log_layer = torch.zeros(32, dtype=torch.float64, requires_grad=True); log_type = torch.zeros(4, dtype=torch.float64, requires_grad=True); bias = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    parameters = [log_global, log_method, log_layer, log_type, bias]; opt = torch.optim.Adam(parameters, lr=a.lr)
    for _ in range(a.steps):
        opt.zero_grad(); coef = torch.exp(log_global + log_method[:,None,None] + log_layer[None,:,None] + log_type[None,None,:]); prediction = bias + (xs * coef).sum((1,2,3)); penalty = sum((p*p).mean() for p in parameters[:-1]); loss = ((prediction[train]-y[train])**2).mean() + a.l2*penalty; loss.backward(); opt.step()
    with torch.no_grad():
        coef = torch.exp(log_global + log_method[:,None,None] + log_layer[None,:,None] + log_type[None,None,:]); pred = (bias + (xs * coef).sum((1,2,3))).tolist()
    rows = [{"policy_id": item["policy_id"], "split": item["split"], "actual_delta_nll": float(y[i]), "predicted_delta_nll": pred[i], "error": pred[i]-float(y[i])} for i,item in enumerate(manifest)]
    out = root / "quality_model" / a.scenario; write_csv(out / "predictions.csv", rows)
    summary = {split: metric([r["actual_delta_nll"] for r in rows if r["split"] == split], [r["predicted_delta_nll"] for r in rows if r["split"] == split]) for split in ("train", "holdout")}
    summary.update({"scenario": a.scenario, "formula": "bias + sum(local_rel_mse*numel*exp(global+method+layer+type))", "phase_weights": dict(phase_weights), "source_local_errors": "artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv"})
    (out / "fit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
