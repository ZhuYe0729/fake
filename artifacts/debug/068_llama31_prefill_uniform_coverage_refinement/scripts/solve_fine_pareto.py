#!/usr/bin/env python3
"""Densify the unchanged 058 solver between the 061 p007/p009 points."""
from __future__ import annotations

import argparse
import csv
import json
import math

from scenario import EXP, KERNEL, METHODS, SOURCE_038, SOURCE_057, SOURCE_061_MODEL, TYPES


STEP = 0.002


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pava(values: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            blocks[-1][0] += right[0]
            blocks[-1][1] += right[1]
    return [total / count for total, count in blocks for _ in range(int(count))]


def corrected(train: list[tuple[float, float]], value: float) -> float:
    pairs = sorted(train)
    xs = [item[0] for item in pairs]
    ys = pava([item[1] for item in pairs])
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for index in range(1, len(xs)):
        if value <= xs[index]:
            ratio = (value - xs[index - 1]) / (xs[index] - xs[index - 1])
            return ys[index - 1] + ratio * (ys[index] - ys[index - 1])
    raise AssertionError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=124, help="First budget in 1e-3 units")
    parser.add_argument("--stop", type=int, default=250, help="Last budget in 1e-3 units")
    parser.add_argument("--stride", type=int, default=6, help="Budget stride in 1e-3 units")
    parser.add_argument("--prefix", default="fine_q")
    parser.add_argument("--output", default="predicted_candidates.csv")
    args = parser.parse_args()
    budgets = tuple(
        value / 1000 for value in range(args.start, args.stop + 1, args.stride)
    )

    model = json.loads((SOURCE_061_MODEL / "reports/quality/model.json").read_text())
    scale = float(model["feature_scale"])

    def coefficient(method_index: int, bucket: int, type_index: int) -> float:
        raw = (
            float(model["global"])
            + float(model["method"][method_index])
            + float(model["bucket"][bucket])
            + float(model["type"][type_index])
        )
        return max(0.0, raw) / scale

    local = {(bucket, fused_type, "dense_bf16"): 0.0 for bucket in range(4) for fused_type in TYPES}
    for method in METHODS[1:]:
        for row in read(SOURCE_057 / "local_errors" / f"prefill_{method}.csv"):
            local[(int(row["layer_bucket"]), row["fused_type"], method)] = float(
                row["output_rel_mse"]
            )

    latency = {
        (row["module_name"], row["kernel"]): float(row["latency_ms"])
        for row in read(SOURCE_038 / "action_support.csv")
        if row["supported"] == "True"
    }
    modules = []
    for layer in range(32):
        for group, fused_type in (
            ("self_attn", "qkv_proj"),
            ("self_attn", "o_proj"),
            ("mlp", "gate_up_proj"),
            ("mlp", "down_proj"),
        ):
            name = f"model.layers.{layer}.{group}.{fused_type}"
            bucket = layer // 8
            options = []
            for method_index, method in enumerate(METHODS):
                quality = local[(bucket, fused_type, method)] * coefficient(
                    method_index, bucket, TYPES.index(fused_type)
                )
                qbin = 0 if method == "dense_bf16" else max(1, math.ceil(quality / STEP))
                options.append((qbin, quality, latency[(name, KERNEL[method])], method))
            modules.append((name, options))

    max_bin = int(max(budgets) / STEP)
    dynamic = [math.inf] * (max_bin + 1)
    dynamic[0] = 0.0
    back = []
    for _, options in modules:
        next_dynamic = [math.inf] * (max_bin + 1)
        choices = [None] * (max_bin + 1)
        for old_state, old_cost in enumerate(dynamic):
            if not math.isfinite(old_cost):
                continue
            for option_index, (qbin, _, latency_ms, _) in enumerate(options):
                new_state = old_state + qbin
                if new_state > max_bin:
                    continue
                candidate = old_cost + latency_ms
                if candidate < next_dynamic[new_state]:
                    next_dynamic[new_state] = candidate
                    choices[new_state] = (old_state, option_index)
        dynamic = next_dynamic
        back.append(choices)

    calibration = read(SOURCE_061_MODEL / "speed/calibration/calibration.csv")
    train = [
        (float(row["raw_predicted_linear_ms"]), float(row["e2e_median_ms"]))
        for row in calibration
        if row["split"] == "train"
    ]
    policy_dir = EXP / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    results = []
    seen = set()
    for budget in budgets:
        limit = int(math.floor(budget / STEP))
        state = min(range(limit + 1), key=lambda index: dynamic[index])
        picks = []
        for index in range(len(modules) - 1, -1, -1):
            item = back[index][state]
            if item is None:
                raise RuntimeError("Unreachable DP state")
            state, option = item
            picks.append(option)
        picks.reverse()
        signature = tuple(picks)
        if signature in seen:
            continue
        seen.add(signature)
        policy_id = f"{args.prefix}{int(round(budget * 1000)):03d}"
        mapping = {
            name: {
                "prefill_method": options[pick][3],
                "decode_method": options[pick][3],
            }
            for (name, options), pick in zip(modules, picks, strict=True)
        }
        raw_latency = sum(
            options[pick][2]
            for (_, options), pick in zip(modules, picks, strict=True)
        )
        predicted_quality = (
            sum(
                options[pick][1]
                for (_, options), pick in zip(modules, picks, strict=True)
            )
            + float(model["bias"])
        )
        policy = {
            "policy_id": policy_id,
            "scenario": "prefill_only",
            "policy_kind": "fine_budget_predicted_pareto",
            "default_prefill_method": "dense_bf16",
            "default_decode_method": "dense_bf16",
            "modules_to_not_convert": ["lm_head"],
            "method_map": mapping,
        }
        (policy_dir / f"{policy_id}.json").write_text(
            json.dumps(policy, indent=2, sort_keys=True) + "\n"
        )
        results.append({
            "policy_id": policy_id,
            "quality_budget": budget,
            "predicted_delta_nll": max(0.0, predicted_quality),
            "raw_predicted_linear_ms": raw_latency,
            "corrected_e2e_prediction_ms": corrected(train, raw_latency),
            **{
                f"count_{method}": sum(
                    item["prefill_method"] == method for item in mapping.values()
                )
                for method in METHODS
            },
        })

    output = EXP / args.output
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(output)


if __name__ == "__main__":
    main()
