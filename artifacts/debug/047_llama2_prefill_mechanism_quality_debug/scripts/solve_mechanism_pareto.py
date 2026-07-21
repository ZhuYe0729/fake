#!/usr/bin/env python3
"""Solve prefill-only policies with the fitted mechanism-aware NLL proxy.

This remains the same multiple-choice knapsack formulation as the original
solver.  The only difference is that a state's quality is evaluated from the
whole policy after reconstruction, so sparse accumulation and sparse/quant
interaction are not incorrectly treated as independent module costs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

from common import CUTLASS, DEBUG, ERRORS, PARTS, ROOT, TYPES

sys.path[:0] = [str(ROOT), str(CUTLASS), str(CUTLASS / "modeling")]
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402

METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
SHAPES = {"qkv_proj": (12288, 4096), "o_proj": (4096, 4096), "gate_up_proj": (22016, 4096), "down_proj": (4096, 11008)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def local_errors(method: str) -> dict[tuple[int, str], float]:
    with ERRORS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for layer in range(32):
        for typ, parts in PARTS.items():
            selected = [float(row["local_rel_mse"]) for row in rows if int(row["layer"]) == layer and row["module_type"] in parts and row["method"] == method]
            values[layer, typ] = sum(selected) / len(selected)
    return values


def modules() -> list[tuple[str, int, str]]:
    return [(f"model.layers.{layer}.{part}.{typ}", layer, typ) for layer in range(32) for part, typ in (("self_attn", "qkv_proj"), ("self_attn", "o_proj"), ("mlp", "gate_up_proj"), ("mlp", "down_proj"))]


def candidates() -> list[list[dict]]:
    predictor = KernelLatencyPredictor(model_root=DEFAULT_MODEL_ROOT, kernels=METHODS)
    tables = {}
    for typ, (n, k) in SHAPES.items():
        linear = {item.kernel: float(item.latency_ms) for item in predictor.predict(16384, n, k).candidates if item.supported and item.latency_ms is not None}
        conversion = {item.conversion: float(item.latency_ms) for item in predictor.predict_conversion(n, k) if item.supported and item.latency_ms is not None}
        tables[typ] = linear, conversion
    groups = []
    for index, (name, layer, typ) in enumerate(modules()):
        linear, conversion = tables[typ]
        rows = []
        for method in METHODS:
            if method not in linear:
                continue
            extra = conversion["canonical_to_cutlass"] if method == "dense_nvfp4" else 0.0
            rows.append({"module_index": index, "module_name": name, "layer": layer, "module_type": typ, "method": method, "latency_ms": linear[method] + extra})
        if not any(row["method"] == "dense_bf16" for row in rows):
            raise RuntimeError(f"missing BF16 candidate for {name}")
        groups.append(rows)
    return groups


def predicted_quality(selected: list[dict], model: dict, q_error: dict, s_error: dict) -> float:
    q = torch.zeros((4, 4), dtype=torch.float64); s = torch.zeros((4, 4), dtype=torch.float64)
    for row in selected:
        bucket, typ = int(row["layer"]) // 8, TYPES.index(row["module_type"])
        if row["method"] in {"dense_nvfp4", "sparse_nvfp4"}:
            q[bucket, typ] += q_error[row["layer"], row["module_type"]]
        if row["method"] in {"sparse_bf16", "sparse_nvfp4"}:
            s[bucket, typ] += s_error[row["layer"], row["module_type"]]
    scale = float(model["feature_scale"])
    q, s = q / scale, s / scale
    qw = torch.tensor(model["quant_weight"], dtype=torch.float64)
    sw = torch.tensor(model["sparse_weight"], dtype=torch.float64)
    a = torch.tensor(model["sparse_accumulation"], dtype=torch.float64)
    c = torch.tensor(model["interaction"], dtype=torch.float64)
    qg, sg = q.sum(1), s.sum(1)
    return float((q * qw).sum() + (s * sw).sum() + (sg.square() * a).sum() + (sg * qg * c).sum())


def solve(groups: list[list[dict]], maximum: float, budget: int, bins: int, model: dict, q_error: dict, s_error: dict) -> tuple[float, float, tuple[int, ...]]:
    # DP uses additive local error merely to retain a compact feasible frontier.
    # Every retained terminal policy is subsequently scored by the full v2 model.
    scale = bins / max(maximum, 1e-12)
    local_q, local_s = local_errors("dense_nvfp4"), local_errors("sparse_bf16")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for rows in groups:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (latency, choices) in states.items():
            for choice, row in enumerate(rows):
                additive = 0.0
                if row["method"] in {"dense_nvfp4", "sparse_nvfp4"}:
                    additive += local_q[row["layer"], row["module_type"]]
                if row["method"] in {"sparse_bf16", "sparse_nvfp4"}:
                    additive += local_s[row["layer"], row["module_type"]]
                total = used + (0 if additive == 0 else max(1, math.ceil(additive * scale)))
                if total > budget:
                    continue
                candidate = latency + float(row["latency_ms"])
                old = next_states.get(total)
                if old is None or candidate < old[0]:
                    next_states[total] = (candidate, choices + (choice,))
        best = math.inf; states = {}
        for key, value in sorted(next_states.items()):
            if value[0] < best:
                states[key] = value; best = value[0]
    feasible = []
    for _, (latency, choices) in states.items():
        selected = [dict(group[index]) for group, index in zip(groups, choices)]
        quality = predicted_quality(selected, model, q_error, s_error)
        if quality <= maximum + 1e-12:
            feasible.append((latency, quality, choices))
    if not feasible:
        raise RuntimeError("no policy satisfies full v2 quality budget")
    latency, quality, choices = min(feasible, key=lambda item: (item[0], item[1]))
    return quality, latency, choices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--budget-bins", type=int, default=4000)
    args = parser.parse_args()
    model = json.loads((DEBUG / "report/model.json").read_text())
    q_error, s_error = local_errors("dense_nvfp4"), local_errors("sparse_bf16")
    groups = candidates()
    all_sparse = [max(group, key=lambda row: float(row["latency_ms"]) * -1 if row["method"] == "sparse_nvfp4" else -1e99) for group in groups]
    maximum = predicted_quality(all_sparse, model, q_error, s_error)
    ratios = [0.0] + [10 ** (-3 + index * 3 / max(args.points - 2, 1)) for index in range(args.points - 1)]
    output, seen = [], set(); policy_dir = DEBUG / "pareto/policies"
    for ratio in ratios:
        quality, latency, choices = solve(groups, ratio * maximum, round(ratio * args.budget_bins), args.budget_bins, model, q_error, s_error)
        selected = [dict(group[index]) for group, index in zip(groups, choices)]
        key = tuple(row["method"] for row in selected)
        if key in seen:
            continue
        seen.add(key); index = len(output)
        policy = {"policy_id": f"v2_{index:03d}", "scenario": "prefill_only", "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16", "modules_to_not_convert": ["lm_head"], "method_map": {row["module_name"]: {"prefill_method": row["method"], "decode_method": "dense_bf16"} for row in selected}}
        path = policy_dir / f"v2_{index:03d}.json"; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        write_csv(path.with_suffix(".csv"), selected)
        counts = {method: sum(row["method"] == method for row in selected) for method in METHODS}
        output.append({"point_id": f"v2_{index:03d}", "quality_budget": ratio * maximum, "v2_predicted_delta_nll": quality, "raw_predicted_linear_ms": latency, **{f"count_{method}": counts[method] for method in METHODS}, "policy_json": str(path)})
    dense = float(output[0]["raw_predicted_linear_ms"])
    for row in output:
        row["raw_linear_speedup_vs_dense"] = dense / float(row["raw_predicted_linear_ms"])
    write_csv(DEBUG / "pareto/pareto_points.csv", output)
    (DEBUG / "pareto/frozen_solver.json").write_text(json.dumps({"quality_model": str(DEBUG / "report/model.json"), "quality_formula": model["proxy"], "solver": "multiple-choice DP pruned by local-error surrogate and checked by full v2 constraint", "quality_metric": "real-vLLM NLL delta vs BF16", "speed_model": "KernelLatencyPredictor roofline model; E2E calibration remains external", "points": len(output)}, indent=2) + "\n")
    print(json.dumps({"points": len(output), "max_v2_delta_nll": maximum, "output": str(DEBUG / "pareto/pareto_points.csv")}, indent=2))


if __name__ == "__main__":
    main()
