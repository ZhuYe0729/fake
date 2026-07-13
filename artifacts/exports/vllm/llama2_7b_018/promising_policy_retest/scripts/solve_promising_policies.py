#!/usr/bin/env python3
"""Solve per-scenario quality-constrained fused hetero policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RETEST_ROOT = SCRIPT_DIR.parent
BASELINE_ROOT = RETEST_ROOT.parent
REPO_ROOT = BASELINE_ROOT.parents[3]
DEFAULT_COST_TABLE = REPO_ROOT / "artifacts/debug/018_llama2_prefill_global_pareto/costs/module_method_candidates.csv"
DEFAULT_PARETO_POINTS = REPO_ROOT / "artifacts/debug/018_llama2_prefill_global_pareto/pareto/pareto_points.csv"
DEFAULT_SCENARIOS = BASELINE_ROOT / "broad_grid_vllm/summary/promising_scenarios_modeling.csv"
DEFAULT_MODELING_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
LAYERS = 32
FUSED_GROUPS = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "self_attn.o_proj": ("self_attn.o_proj",),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    "mlp.down_proj": ("mlp.down_proj",),
}
FUSED_SHAPES = {
    "self_attn.qkv_proj": (12288, 4096),
    "self_attn.o_proj": (4096, 4096),
    "mlp.gate_up_proj": (22016, 4096),
    "mlp.down_proj": (4096, 11008),
}


@dataclass(frozen=True)
class Scenario:
    name: str
    batch: int
    input_seq: int
    output_seq: int
    measured_best_method: str
    measured_best_speedup: str
    measured_hetero_speedup: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-table", type=Path, default=DEFAULT_COST_TABLE)
    parser.add_argument("--pareto-points", type=Path, default=DEFAULT_PARETO_POINTS)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--modeling-root", type=Path, default=DEFAULT_MODELING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RETEST_ROOT / "policies")
    parser.add_argument("--mode", choices=("quality_constrained", "max_speed"), default="quality_constrained")
    parser.add_argument("--policy-prefix", default="policy")
    parser.add_argument("--quality-budget", type=float, default=None)
    parser.add_argument("--quality-budget-point", type=int, default=24)
    parser.add_argument("--budget-bins", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.modeling_root.resolve()))
    from modeling.kernel_predictor import KernelLatencyPredictor  # noqa: WPS433

    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget = args.quality_budget
    if budget is None:
        budget = read_pareto_budget(args.pareto_points, args.quality_budget_point)
    scenarios = read_scenarios(args.scenarios)
    cost_rows = read_cost_rows(args.cost_table)
    groups = build_group_costs(cost_rows)
    max_quality = sum(max(candidate["quality_cost"] for candidate in candidates) for candidates in groups.values())
    scale = args.budget_bins / max_quality if max_quality > 0 else 1.0
    budget_int = int(math.ceil(budget * scale))
    predictor = KernelLatencyPredictor()

    scenario_rows: list[dict[str, Any]] = []
    unique: dict[str, dict[str, Any]] = {}
    latency_cache: dict[tuple[int, int, int, str], float | None] = {}
    for scenario in scenarios:
        policy, summary = solve_scenario(
            predictor=predictor,
            scenario=scenario,
            groups=groups,
            budget=budget,
            budget_int=budget_int,
            scale=scale,
            latency_cache=latency_cache,
            mode=args.mode,
        )
        policy_hash = hash_policy(policy)
        policy_name = f"{args.policy_prefix}_{len(unique):03d}_{policy_hash[:10]}"
        if policy_hash not in unique:
            unique[policy_hash] = {
                "policy_name": policy_name,
                "policy_hash": policy_hash,
                "summary": summary,
                "assignments": policy,
            }
            write_policy(args.output_dir / "unique_policies" / f"{policy_name}.json", unique[policy_hash])
        else:
            policy_name = str(unique[policy_hash]["policy_name"])

        scenario_rows.append(
            {
                "scenario": scenario.name,
                "batch": scenario.batch,
                "input_seq": scenario.input_seq,
                "output_seq": scenario.output_seq,
                "policy_name": policy_name,
                "policy_hash": policy_hash,
                **summary,
                "measured_best_method": scenario.measured_best_method,
                "measured_best_speedup": scenario.measured_best_speedup,
                "measured_hetero_speedup": scenario.measured_hetero_speedup,
            }
        )

    write_csv(args.output_dir / "scenario_policies.csv", scenario_rows)
    write_json(
        args.output_dir / "scenario_policies.json",
        {
            "quality_budget": budget,
            "quality_budget_point": args.quality_budget_point,
            "mode": args.mode,
            "budget_int": budget_int,
            "scale": scale,
            "methods": list(METHODS),
            "scenarios": scenario_rows,
            "unique_policies": list(unique.values()),
        },
    )


def read_pareto_budget(path: Path, point: int) -> float:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["point_index"]) == point:
                return float(row["quality_budget"])
    raise RuntimeError(f"point {point} not found in {path}")


def read_scenarios(path: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            scenarios.append(
                Scenario(
                    name=row["scenario"],
                    batch=int(row["batch"]),
                    input_seq=int(row["input_seq"]),
                    output_seq=int(row["output_seq"]),
                    measured_best_method=row["measured_best_method"],
                    measured_best_speedup=row["measured_best_speedup"],
                    measured_hetero_speedup=row["measured_hetero_speedup"],
                )
            )
    return scenarios


def read_cost_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] not in METHODS:
                continue
            rows.append(row)
    return rows


def build_group_costs(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    by_module: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["layer"]), row["linear_group"], row["method"])
        by_module[key] = row

    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for layer in range(LAYERS):
        for fused_name, source_groups in FUSED_GROUPS.items():
            candidates = []
            for method in METHODS:
                source_rows = [by_module[(layer, group, method)] for group in source_groups]
                candidates.append(
                    {
                        "layer": layer,
                        "fused_name": fused_name,
                        "method": method,
                        "quality_cost": sum(to_float(row["quality_cost"]) for row in source_rows),
                        "source_modules": [row["module_name"] for row in source_rows],
                    }
                )
            groups[(layer, fused_name)] = candidates
    return groups


def solve_scenario(
    *,
    predictor: Any,
    scenario: Scenario,
    groups: dict[tuple[int, str], list[dict[str, Any]]],
    budget: float,
    budget_int: int,
    scale: float,
    latency_cache: dict[tuple[int, int, int, str], float | None],
    mode: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    dense_total = 0.0
    speed_total = 0.0
    enriched: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for key, candidates in groups.items():
        fused_name = key[1]
        n, k = FUSED_SHAPES[fused_name]
        enriched_candidates = []
        for candidate in candidates:
            latency = workload_latency(predictor, scenario, candidate["method"], n, k, latency_cache)
            if latency is None:
                continue
            item = dict(candidate)
            item["latency_cost"] = latency
            item["quality_bin"] = quality_bin(float(item["quality_cost"]), scale)
            enriched_candidates.append(item)
            if candidate["method"] == "dense_bf16":
                dense_total += latency
        if not enriched_candidates:
            raise RuntimeError(f"no candidates for {scenario.name} {key}")
        speed_total += min(float(item["latency_cost"]) for item in enriched_candidates)
        enriched[key] = enriched_candidates

    if mode == "max_speed":
        policy_items = solve_max_speed(enriched)
    else:
        policy_items = solve_budget(enriched, budget_int)
    assignments = {f"model.layers.{layer}.{fused_name}": item["method"] for (layer, fused_name), item in policy_items.items()}
    quality = sum(float(item["quality_cost"]) for item in policy_items.values())
    latency = sum(float(item["latency_cost"]) for item in policy_items.values())
    counts = Counter(item["method"] for item in policy_items.values())
    summary = {
        "quality_budget": f"{budget:.12g}",
        "solve_mode": mode,
        "quality_cost": f"{quality:.12g}",
        "predicted_linear_ms": f"{latency:.6f}",
        "dense_linear_ms": f"{dense_total:.6f}",
        "speed_optimal_linear_ms": f"{speed_total:.6f}",
        "predicted_speedup_vs_dense": f"{dense_total / latency:.6f}" if latency > 0 else "",
        "speed_gap_to_optimal_ms": f"{latency - speed_total:.6f}",
        **{f"count_{method}": counts.get(method, 0) for method in METHODS},
    }
    return assignments, summary


def solve_max_speed(
    groups: dict[tuple[int, str], list[dict[str, Any]]],
) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        key: min(candidates, key=lambda item: (float(item["latency_cost"]), float(item["quality_cost"])))
        for key, candidates in groups.items()
    }


def workload_latency(
    predictor: Any,
    scenario: Scenario,
    method: str,
    n: int,
    k: int,
    latency_cache: dict[tuple[int, int, int, str], float | None],
) -> float | None:
    total = 0.0
    for m, repeats in ((scenario.batch * scenario.input_seq, 1), (scenario.batch, scenario.output_seq)):
        if repeats <= 0:
            continue
        cache_key = (m, n, k, method)
        if cache_key not in latency_cache:
            selection = predictor.predict(m=m, n=n, k=k)
            candidates = {candidate.kernel: candidate for candidate in selection.candidates}
            candidate = candidates.get(method)
            if candidate is None or not candidate.supported or candidate.latency_ms is None:
                latency_cache[cache_key] = None
            else:
                latency_cache[cache_key] = float(candidate.latency_ms)
        latency = latency_cache[cache_key]
        if latency is None:
            return None
        total += latency * repeats
    return total


def solve_budget(
    groups: dict[tuple[int, str], list[dict[str, Any]]],
    budget_int: int,
) -> dict[tuple[int, str], dict[str, Any]]:
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    items = list(groups.items())
    for _key, candidates in items:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used_q, (latency, choices) in states.items():
            for index, candidate in enumerate(candidates):
                q = int(candidate["quality_bin"])
                new_q = used_q + q
                if new_q > budget_int:
                    continue
                new_latency = latency + float(candidate["latency_cost"])
                previous = next_states.get(new_q)
                if previous is None or new_latency < previous[0]:
                    next_states[new_q] = (new_latency, choices + (index,))
        if not next_states:
            raise RuntimeError(f"no feasible state for budget_int={budget_int}")
        states = prune_states(next_states)
    _best_q, (_best_latency, best_choices) = min(states.items(), key=lambda item: (item[1][0], item[0]))
    return {key: candidates[index] for (key, candidates), index in zip(items, best_choices)}


def prune_states(states: dict[int, tuple[float, tuple[int, ...]]]) -> dict[int, tuple[float, tuple[int, ...]]]:
    pruned: dict[int, tuple[float, tuple[int, ...]]] = {}
    best = math.inf
    for q in sorted(states):
        latency, choices = states[q]
        if latency < best:
            pruned[q] = (latency, choices)
            best = latency
    return pruned


def quality_bin(quality: float, scale: float) -> int:
    if quality <= 0:
        return 0
    return max(1, int(math.ceil(quality * scale)))


def hash_policy(assignments: dict[str, str]) -> str:
    payload = json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_policy(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
