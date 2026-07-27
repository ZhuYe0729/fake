#!/usr/bin/env python3
"""Solve Llama2 prefill-only Pareto policies using the real-vLLM NLL proxy."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

from common import CUTLASS, REPO, RUN

sys.path[:0] = [str(REPO), str(CUTLASS), str(CUTLASS / "modeling")]
from modeling.kernel_predictor import KernelLatencyPredictor  # noqa: E402

METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
PREDICTOR_METHOD = {**{method: method for method in METHODS if method != "w4a16_ours"}, "w4a16_ours": "marlin_nvfp4"}
RUNTIME_METHOD = {**{method: method for method in METHODS if method != "w4a16_ours"}, "w4a16_ours": "w4a16_ours"}
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
SHAPES = {"qkv_proj": (12288, 4096), "o_proj": (4096, 4096), "gate_up_proj": (22016, 4096), "down_proj": (4096, 11008)}
PARTS = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "o_proj": ("o_proj",), "gate_up_proj": ("gate_proj", "up_proj"), "down_proj": ("down_proj",)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--budget-bins", type=int, default=2000)
    parser.add_argument("--predictor-root", type=Path, default=RUN / "kernel_profile/modeling")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def local_errors() -> dict[tuple[int, str, str], float]:
    rows = read_csv(RUN / "local_errors/module_method_errors.csv")
    aliases = {"w4a16_ours": "dense_nvfp4"}
    out: dict[tuple[int, str, str], float] = {}
    for layer in range(32):
        for fused_type, parts in PARTS.items():
            for method in METHODS:
                if method == "dense_bf16":
                    out[layer, fused_type, method] = 0.0
                    continue
                values = [float(row["local_rel_mse"]) for row in rows if int(row["layer"]) == layer and row["module_type"] in parts and row["method"] == aliases.get(method, method)]
                out[layer, fused_type, method] = sum(values) / len(values)
    return out


def modules() -> list[tuple[str, int, str]]:
    return [(f"model.layers.{layer}.{part}.{fused_type}", layer, fused_type) for layer in range(32) for part, fused_type in (("self_attn", "qkv_proj"), ("self_attn", "o_proj"), ("mlp", "gate_up_proj"), ("mlp", "down_proj"))]


def latency_tables(predictor_root: Path) -> dict[str, dict[str, float]]:
    predictor = KernelLatencyPredictor(model_root=predictor_root, kernels=("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"))
    tables = {}
    for fused_type, (n, k) in SHAPES.items():
        linear = {item.kernel: float(item.latency_ms) for item in predictor.predict(8 * 2048, n, k).candidates if item.supported and item.latency_ms is not None}
        tables[fused_type] = linear
    return tables


def candidate_groups(coefficients: torch.Tensor, scale: float, errors: dict[tuple[int, str, str], float], predictor_root: Path) -> list[list[dict[str, object]]]:
    tables = latency_tables(predictor_root)
    groups = []
    for module_index, (name, layer, fused_type) in enumerate(modules()):
        latency = tables[fused_type]
        rows = []
        for method in METHODS:
            kernel = PREDICTOR_METHOD[method]
            if kernel not in latency:
                continue
            quality = errors[layer, fused_type, method] * float(coefficients[METHODS.index(method), layer // 8, TYPES.index(fused_type)]) / scale
            rows.append({"module_index": module_index, "module_name": name, "layer": layer, "module_type": fused_type, "prefill_method": method, "decode_method": method, "prefill_runtime": RUNTIME_METHOD[method], "decode_runtime": RUNTIME_METHOD[method], "quality_cost": quality, "latency_ms": latency[kernel]})
        if not any(row["prefill_method"] == "dense_bf16" for row in rows):
            raise RuntimeError(f"missing dense-BF16 candidate for {name}")
        groups.append(rows)
    return groups


def solve(groups: list[list[dict[str, object]]], budget: int, bins: int, max_quality: float) -> tuple[float, float, tuple[int, ...]]:
    bin_scale = bins / max(max_quality, 1e-12)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for rows in groups:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (latency, choices) in states.items():
            for index, row in enumerate(rows):
                quality = float(row["quality_cost"])
                weight = 0 if quality <= 0 else max(1, math.ceil(quality * bin_scale))
                total = used + weight
                if total > budget:
                    continue
                candidate = latency + float(row["latency_ms"])
                previous = next_states.get(total)
                if previous is None or candidate < previous[0]:
                    next_states[total] = (candidate, choices + (index,))
        best = math.inf
        states = {}
        for quality, value in sorted(next_states.items()):
            if value[0] < best:
                states[quality] = value
                best = value[0]
    if not states:
        raise RuntimeError("no feasible DP state")
    quality, (latency, choices) = min(states.items(), key=lambda item: (item[1][0], item[0]))
    return quality / bin_scale, latency, choices


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = RUN
    quality_model = json.loads((root / "reports/quality/model.json").read_text())
    coefficients = torch.tensor(quality_model["coefficients"], dtype=torch.float64)
    groups = candidate_groups(coefficients, float(quality_model["feature_scale"]), local_errors(), args.predictor_root)
    maximum = sum(max(float(row["quality_cost"]) for row in group) for group in groups)
    ratios = [0.0] + [10 ** (-3 + index * 3 / max(args.points - 2, 1)) for index in range(args.points - 1)]
    points, seen = [], set()
    policy_dir = root / "pareto/policies"
    for ratio in ratios:
        quality, latency, choices = solve(groups, round(ratio * args.budget_bins), args.budget_bins, maximum)
        selected = [dict(group[index]) for group, index in zip(groups, choices)]
        key = tuple(row["prefill_method"] for row in selected)
        if key in seen:
            continue
        seen.add(key)
        counts = {method: sum(row["prefill_method"] == method for row in selected) for method in METHODS}
        point_index = len(points)
        policy = {"policy_id": f"point_{point_index:03d}", "scenario": "prefill_only", "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16", "modules_to_not_convert": ["lm_head"], "method_map": {str(row["module_name"]): {"prefill_method": str(row["prefill_runtime"]), "decode_method": str(row["decode_runtime"])} for row in selected}}
        policy_path = policy_dir / f"point_{point_index:03d}.json"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        write_csv(policy_path.with_suffix(".csv"), selected)
        points.append({"point_index": point_index, "quality_ratio": ratio, "predicted_delta_nll": quality, "raw_predicted_linear_ms": latency, "raw_linear_speedup_vs_dense": 0.0, **{f"prefill_count_{method}": counts[method] for method in METHODS}, "policy_json": str(policy_path)})
    dense = points[0]["raw_predicted_linear_ms"]
    for point in points:
        point["raw_linear_speedup_vs_dense"] = float(dense) / float(point["raw_predicted_linear_ms"])
    write_csv(root / "pareto/pareto_points.csv", points)
    (root / "pareto/frozen_quality_model.json").write_text(json.dumps({"label_backend": "real-vLLM prompt-logprob", "quality_model": str(root / "reports/quality/model.json"), "quality_intercept_excluded_from_optimizer": True, "quality_metric": "predicted delta NLL vs dense BF16", "speed_model": "064 Pro 6000 KernelLatencyPredictor module-forward only", "one_time_weight_conversion_excluded": True, "predictor_root": str(args.predictor_root.resolve()), "points": len(points)}, indent=2) + "\n")
    print(json.dumps({"points": len(points), "max_predicted_delta_nll": maximum, "output": str(root / "pareto/pareto_points.csv")}, indent=2))


if __name__ == "__main__":
    main()
