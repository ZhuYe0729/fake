#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from common_fakevlm_pareto import DEBUG_ROOT, METHODS, f, read_csv, read_json, write_csv, write_json


LOSS_DEFINITION = "assistant_answer_token_nll_v2_active_prefix_aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit FakeVLM multiplicative quality-cost coefficients from measured policy NLL.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--quality-csv", type=Path, default=None)
    parser.add_argument("--metric", default="output_rel_mse")
    parser.add_argument("--target", choices=["nll_delta"], default="nll_delta")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_csv = args.quality_csv or args.output_root / "quality" / "stratified_loss.csv"
    quality_rows = read_csv(quality_csv)
    invalid_definitions = sorted({row.get("loss_definition", "") for row in quality_rows if row.get("loss_definition", "") != LOSS_DEFINITION})
    if invalid_definitions:
        raise RuntimeError(
            f"quality rows use invalid loss definitions {invalid_definitions}; expected {LOSS_DEFINITION}. "
            "Regenerate policy NLL after fixing left-padding and image-token label alignment."
        )
    local_rows = read_csv(args.output_root / "sensitivity" / "module_method_local_errors.csv")
    local = {(row["module_name"], row["method"]): row for row in local_rows}
    dense_nll = baseline_nll(quality_rows)
    samples = []
    layers = sorted({str(int(f(row, "layer"))) for row in local_rows})
    types = sorted({row["module_type"] for row in local_rows})
    method_list = [method for method in METHODS if method != "dense_bf16"]
    for qrow in quality_rows:
        policy_path = Path(qrow["policy_json"])
        policy = read_json(policy_path)
        raw_delta = f(qrow, "nll", default=dense_nll) - dense_nll
        target = max(0.0, raw_delta)
        features = sample_features(policy, local, layers, types, method_list, args.metric)
        samples.append({"quality_row": qrow, "target": target, "raw_delta": raw_delta, "features": features})
    if not samples:
        raise RuntimeError(f"no quality rows in {quality_csv}")

    fit = fit_model(samples, layers, types, method_list, steps=args.steps, lr=args.lr, l2=args.l2)
    predictions = []
    for sample in samples:
        pred = predict(sample["features"], fit, layers, types, method_list)
        row = dict(sample["quality_row"])
        row["raw_nll_delta_vs_dense"] = sample["raw_delta"]
        row["target_nll_delta"] = sample["target"]
        row["predicted_quality_cost"] = pred
        row["residual"] = pred - sample["target"]
        predictions.append(row)

    coefficients = {}
    for method in method_list:
        coefficients[method] = {
            "final_layer_type": {
                "variant": "final_layer_type",
                "target": args.target,
                "local_error_metric": args.metric,
                "global_coef": fit["global"][method],
                "layer_coef": {layer: fit["layer"][(method, layer)] for layer in layers},
                "type_coef": {typ: fit["type"][(method, typ)] for typ in types},
            }
        }
    write_json(args.output_root / "global_coefficients" / "proxy_ablation_coefficients.json", coefficients)
    write_csv(args.output_root / "global_coefficients" / "proxy_ablation_predictions.csv", predictions)
    write_json(
        args.output_root / "global_coefficients" / "proxy_ablation_metadata.json",
        {
            "quality_csv": str(quality_csv),
            "rows": len(samples),
            "dense_nll": dense_nll,
            "methods": method_list,
            "layers": layers,
            "types": types,
            "steps": args.steps,
            "lr": args.lr,
            "rmse": rmse(predictions),
        },
    )
    print(f"wrote coefficients for {len(method_list)} methods; rmse={rmse(predictions):.6g}")


def baseline_nll(rows: list[dict[str, Any]]) -> float:
    dense = [row for row in rows if "dense" in row.get("label", "") or row.get("policy_index") in {"0", "0.0"}]
    if dense:
        return f(dense[0], "nll")
    return min(f(row, "nll") for row in rows if row.get("nll", "") != "")


def sample_features(
    policy: dict[str, Any],
    local: dict[tuple[str, str], dict[str, Any]],
    layers: list[str],
    types: list[str],
    methods: list[str],
    metric: str,
) -> dict[tuple[str, str, str], float]:
    out = {(method, layer, typ): 0.0 for method in methods for layer in layers for typ in types}
    for item in policy["modules"]:
        method = item.get("selected_method") or item.get("backend")
        if method == "dense_bf16":
            continue
        row = local[(item.get("module_name") or item["name"], method)]
        key = (method, str(int(f(row, "layer"))), row["module_type"])
        out[key] += f(row, metric)
    return out


def fit_model(samples: list[dict[str, Any]], layers: list[str], types: list[str], methods: list[str], *, steps: int, lr: float, l2: float) -> dict[str, Any]:
    raw_global = torch.nn.Parameter(torch.zeros(len(methods)))
    raw_layer = torch.nn.Parameter(torch.zeros(len(methods), len(layers)))
    raw_type = torch.nn.Parameter(torch.zeros(len(methods), len(types)))
    params = [raw_global, raw_layer, raw_type]
    opt = torch.optim.Adam(params, lr=lr)
    targets = torch.tensor([sample["target"] for sample in samples], dtype=torch.float32)
    feature_t = torch.zeros(len(samples), len(methods), len(layers), len(types), dtype=torch.float32)
    method_index = {method: i for i, method in enumerate(methods)}
    layer_index = {layer: i for i, layer in enumerate(layers)}
    type_index = {typ: i for i, typ in enumerate(types)}
    for sample_i, sample in enumerate(samples):
        for (method, layer, typ), value in sample["features"].items():
            if value:
                feature_t[sample_i, method_index[method], layer_index[layer], type_index[typ]] = float(value)
    for _ in range(steps):
        coef_t = (
            positive(raw_global).view(1, len(methods), 1, 1)
            * positive(raw_layer).view(1, len(methods), len(layers), 1)
            * positive(raw_type).view(1, len(methods), 1, len(types))
        )
        pred_t = (feature_t * coef_t).sum(dim=(1, 2, 3))
        reg = sum(param.pow(2).sum() for param in params)
        loss = torch.mean((pred_t - targets).pow(2)) + l2 * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    global_fit = positive(raw_global).detach()
    layer_fit = positive(raw_layer).detach()
    type_fit = positive(raw_type).detach()
    return {
        "global": {method: float(global_fit[method_index[method]]) for method in methods},
        "layer": {(method, layer): float(layer_fit[method_index[method], layer_index[layer]]) for method in methods for layer in layers},
        "type": {(method, typ): float(type_fit[method_index[method], type_index[typ]]) for method in methods for typ in types},
    }


def positive(param: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(param) + 1e-8


def predict(features: dict[tuple[str, str, str], float], fit: dict[str, Any], layers: list[str], types: list[str], methods: list[str]) -> float:
    total = 0.0
    for method in methods:
        for layer in layers:
            for typ in types:
                total += (
                    features[(method, layer, typ)]
                    * fit["global"][method]
                    * fit["layer"][(method, layer)]
                    * fit["type"][(method, typ)]
                )
    return total


def rmse(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    err = [(f(row, "predicted_quality_cost") - f(row, "target_nll_delta")) ** 2 for row in rows]
    return math.sqrt(sum(err) / len(err))


if __name__ == "__main__":
    main()
