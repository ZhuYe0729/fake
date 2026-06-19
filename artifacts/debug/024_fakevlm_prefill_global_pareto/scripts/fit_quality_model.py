#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from common_fakevlm_pareto import DEBUG_ROOT, METHODS, f, read_csv, read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit FakeVLM multiplicative quality-cost coefficients from measured policies.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--quality-csv", type=Path, default=None)
    parser.add_argument("--metric", default="output_rel_mse")
    parser.add_argument("--target", choices=["accuracy_drop"], default="accuracy_drop")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_csv = args.quality_csv or args.output_root / "quality" / "stratified_quality.csv"
    quality_rows = read_csv(quality_csv)
    local_rows = read_csv(args.output_root / "sensitivity" / "module_method_local_errors.csv")
    local = {(row["module_name"], row["method"]): row for row in local_rows}
    dense_acc = baseline_accuracy(quality_rows)
    samples = []
    layers = sorted({str(int(f(row, "layer"))) for row in local_rows})
    types = sorted({row["module_type"] for row in local_rows})
    method_list = [method for method in METHODS if method != "dense_bf16"]
    for qrow in quality_rows:
        policy_path = Path(qrow["policy_json"])
        policy = read_json(policy_path)
        target = max(0.0, dense_acc - f(qrow, "global_accuracy"))
        features = sample_features(policy, local, layers, types, method_list, args.metric)
        samples.append({"quality_row": qrow, "target": target, "features": features})
    if not samples:
        raise RuntimeError(f"no quality rows in {quality_csv}")

    fit = fit_model(samples, layers, types, method_list, steps=args.steps, lr=args.lr, l2=args.l2)
    predictions = []
    for sample in samples:
        pred = predict(sample["features"], fit, layers, types, method_list)
        row = dict(sample["quality_row"])
        row["target_accuracy_drop"] = sample["target"]
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
            "dense_accuracy": dense_acc,
            "methods": method_list,
            "layers": layers,
            "types": types,
            "steps": args.steps,
            "lr": args.lr,
            "rmse": rmse(predictions),
        },
    )
    print(f"wrote coefficients for {len(method_list)} methods; rmse={rmse(predictions):.6g}")


def baseline_accuracy(rows: list[dict[str, Any]]) -> float:
    dense = [row for row in rows if "dense" in row.get("label", "") or row.get("policy_index") in {"0", "0.0"}]
    if dense:
        return f(dense[0], "global_accuracy")
    return max(f(row, "global_accuracy") for row in rows)


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
    raw_global = {method: torch.nn.Parameter(torch.tensor(0.0)) for method in methods}
    raw_layer = {(method, layer): torch.nn.Parameter(torch.tensor(0.0)) for method in methods for layer in layers}
    raw_type = {(method, typ): torch.nn.Parameter(torch.tensor(0.0)) for method in methods for typ in types}
    params = [*raw_global.values(), *raw_layer.values(), *raw_type.values()]
    opt = torch.optim.Adam(params, lr=lr)
    targets = torch.tensor([sample["target"] for sample in samples], dtype=torch.float32)
    for _ in range(steps):
        preds = []
        for sample in samples:
            pred = torch.tensor(0.0)
            for method in methods:
                g = positive(raw_global[method])
                for layer in layers:
                    lcoef = positive(raw_layer[(method, layer)])
                    for typ in types:
                        value = sample["features"][(method, layer, typ)]
                        if value:
                            pred = pred + float(value) * g * lcoef * positive(raw_type[(method, typ)])
            preds.append(pred)
        pred_t = torch.stack(preds)
        reg = sum(param.pow(2) for param in params)
        loss = torch.mean((pred_t - targets).pow(2)) + l2 * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {
        "global": {method: float(positive(raw_global[method]).detach()) for method in methods},
        "layer": {(method, layer): float(positive(raw_layer[(method, layer)]).detach()) for method in methods for layer in layers},
        "type": {(method, typ): float(positive(raw_type[(method, typ)]).detach()) for method in methods for typ in types},
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
    err = [(f(row, "predicted_quality_cost") - f(row, "target_accuracy_drop")) ** 2 for row in rows]
    return math.sqrt(sum(err) / len(err))


if __name__ == "__main__":
    main()
