#!/usr/bin/env python3
"""Solve canonical phase policies with the established roofline speed model."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scenario import BATCH, INPUT_TOKENS, DECODE_STEPS, EXP
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
KERNEL = {**{method: method for method in METHODS[:-1]}, "w4a16_ours": "marlin_nvfp4"}
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
PHASES = ("prefill", "decode")
STEP = 0.002


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def local_errors() -> dict[tuple[str, int, str, str], float]:
    result = {}
    for phase in PHASES:
        for method in METHODS:
            if method == "dense_bf16":
                for bucket in range(4):
                    for typ in TYPES: result[phase, bucket, typ, method] = 0.0
            else:
                for row in read(EXP / "local_errors" / f"{phase}_{method}.csv"):
                    result[phase, int(row["layer_bucket"]), row["fused_type"], method] = float(row["output_rel_mse"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXP / "pareto")
    parser.add_argument("--policy-prefix", default="point_")
    parser.add_argument("--target-raw-speedups", default="",
                        help="Comma-separated roofline speedup targets; solve minimum proxy loss at each target.")
    args = parser.parse_args()
    model = json.loads((EXP / "reports/quality/model.json").read_text())
    coefficients = model["coefficients"]
    errors = local_errors()
    actions = {(row["phase"], row["module_name"], row["kernel"]): row
               for row in read(EXP / "speed/action_support.csv")}
    groups = []
    for layer in range(32):
        for group, typ in (("self_attn", "qkv_proj"), ("self_attn", "o_proj"), ("mlp", "gate_up_proj"), ("mlp", "down_proj")):
            name, bucket = f"model.layers.{layer}.{group}.{typ}", layer // 8
            options = []
            for pre in METHODS:
                for dec in METHODS:
                    pre_action, dec_action = actions["prefill", name, KERNEL[pre]], actions["decode", name, KERNEL[dec]]
                    if pre_action["supported"] != "True" or dec_action["supported"] != "True": continue
                    q = sum(errors[phase, bucket, typ, method] * coefficients[phase_index][METHODS.index(method)][bucket][TYPES.index(typ)] / model["feature_scale"]
                            for phase_index, (phase, method) in enumerate((("prefill", pre), ("decode", dec))))
                    latency = float(pre_action["latency_ms"]) + DECODE_STEPS * float(dec_action["latency_ms"])
                    options.append((0 if q <= 0 else max(1, math.ceil(q / STEP)), q, latency, pre, dec))
            if not options: raise RuntimeError(f"no legal actions for {name}")
            groups.append((name, options))
    maximum = sum(max(option[0] for option in options) for _, options in groups)
    dp = [math.inf] * (maximum + 1); dp[0] = 0.0; backs = []
    for _, options in groups:
        nxt = [math.inf] * (maximum + 1); back = [None] * (maximum + 1)
        for old, cost in enumerate(dp):
            if not math.isfinite(cost): continue
            for index, option in enumerate(options):
                new = min(maximum, old + option[0]); candidate = cost + option[2]
                if candidate < nxt[new]: nxt[new], back[new] = candidate, (old, index)
        dp, backs = nxt, backs + [back]
    budgets = [0.0] + [10 ** (-2.3 + index * (math.log10(2.0) + 2.3) / 23)
                       for index in range(24)]
    output, policy_dir = args.output_dir, args.output_dir / "policies"; policy_dir.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    # The quality model is a proxy, so some non-BF16 cells can receive an exact
    # zero score.  Keep the physical dense-BF16 policy as the explicit speed
    # reference instead of letting such a cell redefine the baseline.
    baseline = []
    for _, options in groups:
        baseline.append(next(option for option in options
                             if option[3] == "dense_bf16" and option[4] == "dense_bf16"))
    baseline_raw = sum(value[2] for value in baseline)
    baseline_map = {name: {"prefill_method": "dense_bf16", "decode_method": "dense_bf16"}
                    for name, _ in groups}
    baseline_id = f"{args.policy_prefix}000"
    (policy_dir / f"{baseline_id}.json").write_text(json.dumps({"policy_id": baseline_id, "scenario": "prefill_decode",
        "policy_kind": "dense_bf16_reference", "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16",
        "modules_to_not_convert": ["lm_head"], "method_map": baseline_map}, indent=2, sort_keys=True) + "\n")
    rows.append({"policy_id": baseline_id, "quality_budget": 0.0, "predicted_delta_nll": 0.0,
                 "raw_predicted_linear_ms": baseline_raw,
                 **{f"prefill_count_{method}": 128 if method == "dense_bf16" else 0 for method in METHODS},
                 **{f"decode_count_{method}": 128 if method == "dense_bf16" else 0 for method in METHODS}})
    seen.add(tuple(next(index for index, option in enumerate(options)
                        if option[3] == "dense_bf16" and option[4] == "dense_bf16") for _, options in groups))
    if args.target_raw_speedups:
        selections = []
        for target in (float(value) for value in args.target_raw_speedups.split(",") if value.strip()):
            cap = baseline_raw / target
            states = [state for state, latency in enumerate(dp) if latency <= cap]
            if not states: continue
            selections.append((target, min(states)))
    else:
        selections = []
        for budget in budgets:
            limit = min(maximum, int(budget / STEP))
            selections.append((budget, min(range(limit + 1), key=lambda item: dp[item])))
    for budget, state in selections:
        picked = []
        for back in reversed(backs):
            previous, option = back[state]; picked.append(option); state = previous
        picked.reverse(); signature = tuple(picked)
        if signature in seen: continue
        seen.add(signature)
        selected = [options[index] for (_, options), index in zip(groups, picked)]
        policy_id = f"{args.policy_prefix}{len(rows):03d}"
        method_map = {name: {"prefill_method": value[3], "decode_method": value[4]}
                      for (name, _), value in zip(groups, selected)}
        (policy_dir / f"{policy_id}.json").write_text(json.dumps({"policy_id": policy_id, "scenario": "prefill_decode",
            "policy_kind": "predicted_pareto", "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16",
            "modules_to_not_convert": ["lm_head"], "method_map": method_map}, indent=2, sort_keys=True) + "\n")
        raw = sum(value[2] for value in selected)
        rows.append({"policy_id": policy_id, "quality_budget": budget, "predicted_delta_nll": sum(value[1] for value in selected),
                     "raw_predicted_linear_ms": raw,
                     **{f"prefill_count_{method}": sum(value[3] == method for value in selected) for method in METHODS},
                     **{f"decode_count_{method}": sum(value[4] == method for value in selected) for method in METHODS}})
    dense_raw = baseline_raw
    for row in rows:
        row["raw_speedup_vs_dense"] = dense_raw / row["raw_predicted_linear_ms"]
    with (output / "predicted_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "metadata.json").write_text(json.dumps({"quality": "066 Llama3.1 phase-aware canonical real-vLLM ΔNLL proxy",
        "speed": "KernelLatencyPredictor roofline kernel-cost sum; final E2E speed is measured during closure",
        "raw_formula": f"sum prefill(M={BATCH * INPUT_TOKENS}) + {DECODE_STEPS} * sum decode(M={BATCH})", "decode_sparse_nvfp4": "excluded by action-support audit",
        "selection": "target raw speedup" if args.target_raw_speedups else "quality budget",
        "status": "predicted screening; final E2E/NLL closure required"}, indent=2) + "\n")


if __name__ == "__main__": main()
