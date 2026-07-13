#!/usr/bin/env python3
"""Analyze Llama2 vLLM prefill-only workloads for hetero compression."""

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
SEARCH_ROOT = SCRIPT_DIR.parent
BASELINE_ROOT = SEARCH_ROOT.parent
REPO_ROOT = BASELINE_ROOT.parents[3]
MODELING_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
DEFAULT_COST_TABLE = REPO_ROOT / "artifacts/debug/018_llama2_prefill_global_pareto/costs/module_method_candidates.csv"
DEFAULT_PARETO_POINTS = REPO_ROOT / "artifacts/debug/018_llama2_prefill_global_pareto/pareto/pareto_points.csv"
DEFAULT_RETEST_SUMMARY = BASELINE_ROOT / "promising_policy_retest/summary/promising_policy_retest_summary.csv"
DEFAULT_OUTPUT_DIR = SEARCH_ROOT / "summary"

HETERO_METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
UNIFORM_METHODS = ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")
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
    batch: int
    input_seq: int

    @property
    def name(self) -> str:
        return f"b{self.batch}_in{self.input_seq}_out1"

    @property
    def prefill_m(self) -> int:
        return self.batch * self.input_seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-table", type=Path, default=DEFAULT_COST_TABLE)
    parser.add_argument("--pareto-points", type=Path, default=DEFAULT_PARETO_POINTS)
    parser.add_argument("--retest-summary", type=Path, default=DEFAULT_RETEST_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality-budget-point", type=int, default=24)
    parser.add_argument("--budget-bins", type=int, default=2000)
    parser.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--input-seqs", default="128,256,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(MODELING_ROOT.resolve()))
    from modeling.kernel_predictor import KernelLatencyPredictor  # noqa: WPS433

    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget = read_pareto_budget(args.pareto_points, args.quality_budget_point)
    groups = build_group_costs(read_cost_rows(args.cost_table))
    max_quality = sum(max(candidate["quality_cost"] for candidate in candidates) for candidates in groups.values())
    scale = args.budget_bins / max_quality if max_quality > 0 else 1.0
    budget_int = int(math.ceil(budget * scale))
    predictor = KernelLatencyPredictor()
    latency_cache: dict[tuple[int, int, int, str], float | None] = {}

    scenarios = [
        Scenario(batch=batch, input_seq=input_seq)
        for batch in parse_ints(args.batches)
        for input_seq in parse_ints(args.input_seqs)
    ]
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        row, details, policies = analyze_scenario(
            predictor=predictor,
            scenario=scenario,
            groups=groups,
            budget=budget,
            budget_int=budget_int,
            scale=scale,
            latency_cache=latency_cache,
        )
        rows.append(row)
        detail_rows.extend(details)
        policy_rows.extend(policies)

    rows.sort(key=lambda row: (float(row["optimized_vs_best_uniform"]), float(row["max_speed_vs_best_uniform"])), reverse=True)
    measured_rows = read_existing_measured_prefill(args.retest_summary)
    top_rows = rows[: args.top_k]

    write_csv(args.output_dir / "prefill_only_prediction_candidates.csv", rows)
    write_csv(args.output_dir / "prefill_only_prediction_details.csv", detail_rows)
    write_csv(args.output_dir / "prefill_only_policy_rows.csv", policy_rows)
    write_csv(args.output_dir / "existing_prefill_only_measured.csv", measured_rows)
    write_markdown(args.output_dir / "prefill_only_workload_search.md", rows, measured_rows, top_rows, budget)
    write_retest_files(args.output_dir, top_rows)


def analyze_scenario(
    *,
    predictor: Any,
    scenario: Scenario,
    groups: dict[tuple[int, str], list[dict[str, Any]]],
    budget: float,
    budget_int: int,
    scale: float,
    latency_cache: dict[tuple[int, int, int, str], float | None],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    enriched: dict[tuple[int, str], list[dict[str, Any]]] = {}
    method_totals = {"dense_bf16": 0.0, **{method: 0.0 for method in UNIFORM_METHODS}}
    detail_rows: list[dict[str, Any]] = []
    for key, candidates in groups.items():
        _layer, fused_name = key
        n, k = FUSED_SHAPES[fused_name]
        enriched_candidates = []
        call_latencies: dict[str, float] = {}
        for method in ("dense_bf16", *UNIFORM_METHODS):
            latency = predict_latency(predictor, scenario.prefill_m, n, k, method, latency_cache)
            if latency is not None:
                call_latencies[method] = latency
        for method, latency in call_latencies.items():
            method_totals[method] += latency
        for candidate in candidates:
            latency = call_latencies.get(candidate["method"])
            if latency is None:
                continue
            item = dict(candidate)
            item["latency_cost"] = latency
            item["quality_bin"] = quality_bin(float(item["quality_cost"]), scale)
            enriched_candidates.append(item)
        enriched[key] = enriched_candidates
        if key[0] == 0:
            best_uniform = min(UNIFORM_METHODS, key=lambda method: call_latencies.get(method, math.inf))
            best_hetero = min(HETERO_METHODS, key=lambda method: call_latencies.get(method, math.inf))
            detail_rows.append(
                {
                    "scenario": scenario.name,
                    "prefill_m": scenario.prefill_m,
                    "fused_name": fused_name,
                    "best_uniform": best_uniform,
                    "best_hetero_candidate": best_hetero,
                    **{f"{method}_single_call_ms": fmt(call_latencies.get(method)) for method in ("dense_bf16", *UNIFORM_METHODS)},
                }
            )

    best_uniform_method = min(UNIFORM_METHODS, key=lambda method: method_totals[method])
    best_uniform_ms = method_totals[best_uniform_method]
    dense_ms = method_totals["dense_bf16"]
    optimized_items = solve_budget(enriched, budget_int)
    max_speed_items = solve_max_speed(enriched)
    optimized = summarize_policy(optimized_items)
    max_speed = summarize_policy(max_speed_items)
    row = {
        "scenario": scenario.name,
        "batch": scenario.batch,
        "input_seq": scenario.input_seq,
        "output_seq": 1,
        "prefill_m": scenario.prefill_m,
        "quality_budget": f"{budget:.12g}",
        "dense_bf16_prefill_ms": fmt(dense_ms),
        **{f"uniform_{method}_prefill_ms": fmt(method_totals[method]) for method in UNIFORM_METHODS},
        "best_uniform_method": best_uniform_method,
        "best_uniform_prefill_ms": fmt(best_uniform_ms),
        "optimized_prefill_ms": fmt(optimized["latency"]),
        "optimized_quality_cost": f"{optimized['quality']:.12g}",
        "optimized_speedup_vs_dense": fmt(dense_ms / optimized["latency"]),
        "optimized_vs_best_uniform": fmt(best_uniform_ms / optimized["latency"]),
        "optimized_counts": counter_text(optimized["counts"]),
        "optimized_policy_hash": policy_hash(optimized_items),
        "max_speed_prefill_ms": fmt(max_speed["latency"]),
        "max_speed_quality_cost": f"{max_speed['quality']:.12g}",
        "max_speed_speedup_vs_dense": fmt(dense_ms / max_speed["latency"]),
        "max_speed_vs_best_uniform": fmt(best_uniform_ms / max_speed["latency"]),
        "max_speed_counts": counter_text(max_speed["counts"]),
        "max_speed_policy_hash": policy_hash(max_speed_items),
    }
    policy_rows = [
        policy_row(scenario, "optimized", optimized_items),
        policy_row(scenario, "max_speed", max_speed_items),
    ]
    return row, detail_rows, policy_rows


def read_pareto_budget(path: Path, point: int) -> float:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["point_index"]) == point:
                return float(row["quality_budget"])
    raise RuntimeError(f"point {point} not found in {path}")


def read_cost_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["method"] in HETERO_METHODS]


def build_group_costs(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    by_module = {(int(row["layer"]), row["linear_group"], row["method"]): row for row in rows}
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for layer in range(LAYERS):
        for fused_name, source_groups in FUSED_GROUPS.items():
            candidates = []
            for method in HETERO_METHODS:
                source_rows = [by_module[(layer, group, method)] for group in source_groups]
                candidates.append(
                    {
                        "layer": layer,
                        "fused_name": fused_name,
                        "method": method,
                        "quality_cost": sum(to_float(row["quality_cost"]) for row in source_rows),
                    }
                )
            groups[(layer, fused_name)] = candidates
    return groups


def predict_latency(
    predictor: Any,
    m: int,
    n: int,
    k: int,
    method: str,
    cache: dict[tuple[int, int, int, str], float | None],
) -> float | None:
    key = (m, n, k, method)
    if key not in cache:
        selection = predictor.predict(m=m, n=n, k=k)
        candidates = {candidate.kernel: candidate for candidate in selection.candidates}
        candidate = candidates.get(method)
        if candidate is None or not candidate.supported or candidate.latency_ms is None:
            cache[key] = None
        else:
            cache[key] = float(candidate.latency_ms)
    return cache[key]


def solve_max_speed(groups: dict[tuple[int, str], list[dict[str, Any]]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {key: min(candidates, key=lambda item: (float(item["latency_cost"]), float(item["quality_cost"]))) for key, candidates in groups.items()}


def solve_budget(groups: dict[tuple[int, str], list[dict[str, Any]]], budget_int: int) -> dict[tuple[int, str], dict[str, Any]]:
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    items = list(groups.items())
    for _key, candidates in items:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used_q, (latency, choices) in states.items():
            for index, candidate in enumerate(candidates):
                new_q = used_q + int(candidate["quality_bin"])
                if new_q > budget_int:
                    continue
                new_latency = latency + float(candidate["latency_cost"])
                previous = next_states.get(new_q)
                if previous is None or new_latency < previous[0]:
                    next_states[new_q] = (new_latency, choices + (index,))
        states = prune_states(next_states)
    _best_q, (_best_latency, choices) = min(states.items(), key=lambda item: (item[1][0], item[0]))
    return {key: candidates[index] for (key, candidates), index in zip(items, choices)}


def prune_states(states: dict[int, tuple[float, tuple[int, ...]]]) -> dict[int, tuple[float, tuple[int, ...]]]:
    pruned: dict[int, tuple[float, tuple[int, ...]]] = {}
    best = math.inf
    for q in sorted(states):
        latency, choices = states[q]
        if latency < best:
            pruned[q] = (latency, choices)
            best = latency
    return pruned


def summarize_policy(items: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    return {
        "latency": sum(float(item["latency_cost"]) for item in items.values()),
        "quality": sum(float(item["quality_cost"]) for item in items.values()),
        "counts": Counter(item["method"] for item in items.values()),
    }


def policy_hash(items: dict[tuple[int, str], dict[str, Any]]) -> str:
    assignments = {f"model.layers.{layer}.{fused_name}": item["method"] for (layer, fused_name), item in items.items()}
    payload = json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:10]


def policy_row(scenario: Scenario, mode: str, items: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": scenario.name,
        "batch": scenario.batch,
        "input_seq": scenario.input_seq,
        "output_seq": 1,
        "mode": mode,
        "policy_hash": policy_hash(items),
    }
    for (layer, fused_name), item in items.items():
        row[f"model.layers.{layer}.{fused_name}"] = item["method"]
    return row


def read_existing_measured_prefill(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("output_seq") != "1":
                continue
            rows.append(
                {
                    "scenario": row["scenario"],
                    "batch": row["batch"],
                    "input_seq": row["input_seq"],
                    "best_single_method": row["best_single_method"],
                    "optimized_speedup_vs_best_single": row["optimized_speedup_vs_best_single"],
                    "optimized_arc_acc_norm": row["optimized_arc_acc_norm"],
                    "max_speed_speedup_vs_best_single": row["max_speed_speedup_vs_best_single"],
                    "max_speed_arc_acc_norm": row["max_speed_arc_acc_norm"],
                }
            )
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]], measured_rows: list[dict[str, Any]], top_rows: list[dict[str, Any]], budget: float) -> None:
    lines = [
        "# Llama2 vLLM Prefill-Only Workload Search",
        "",
        f"Quality budget uses Pareto point P024: `{budget:.12g}`.",
        "",
        "## Existing measured prefill-only proxy",
        "",
        "| scenario | best_single | optimized_vs_best_single | optimized_acc_norm | max_speed_vs_best_single | max_speed_acc_norm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in measured_rows:
        lines.append(
            f"| {row['scenario']} | {row['best_single_method']} | {row['optimized_speedup_vs_best_single']} | "
            f"{row['optimized_arc_acc_norm']} | {row['max_speed_speedup_vs_best_single']} | {row['max_speed_arc_acc_norm']} |"
        )
    lines.extend(
        [
            "",
            "## Top pure-prefill predicted candidates",
            "",
            "| scenario | prefill_m | best_uniform | optimized_vs_best_uniform | optimized_counts | max_speed_vs_best_uniform | max_speed_counts |",
            "|---|---:|---|---:|---|---:|---|",
        ]
    )
    for row in top_rows:
        lines.append(
            f"| {row['scenario']} | {row['prefill_m']} | {row['best_uniform_method']} | "
            f"{row['optimized_vs_best_uniform']} | {row['optimized_counts']} | "
            f"{row['max_speed_vs_best_uniform']} | {row['max_speed_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `optimized` is the P024 quality-constrained fused hetero policy.",
            "- `max_speed` is the unconstrained fastest fused hetero policy and is useful as a speed upper bound, but may have unacceptable accuracy loss.",
            "- Pure-prefill prediction excludes the decode `M=batch` call that vLLM `output_seq=1` still executes.",
            "- Final claims should use the generated focused retest commands and full quality results.",
            "",
            "## Generated files",
            "",
            "- `prefill_only_prediction_candidates.csv`",
            "- `prefill_only_prediction_details.csv`",
            "- `prefill_only_policy_rows.csv`",
            "- `existing_prefill_only_measured.csv`",
            "- `focused_retest_scenarios.csv`",
            "- `run_focused_retest.sh`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_retest_files(output_dir: Path, top_rows: list[dict[str, Any]]) -> None:
    scenario_rows = [
        {
            "scenario": row["scenario"],
            "batch": row["batch"],
            "input_seq": row["input_seq"],
            "output_seq": 1,
            "measured_best_method": row["best_uniform_method"],
            "measured_best_speedup": "",
            "measured_hetero_speedup": "",
        }
        for row in top_rows
    ]
    write_csv(output_dir / "focused_retest_scenarios.csv", scenario_rows)
    scenarios = ",".join(str(row["scenario"]) for row in top_rows)
    batches = ",".join(dict.fromkeys(str(row["batch"]) for row in top_rows))
    input_seqs = ",".join(dict.fromkeys(str(row["input_seq"]) for row in top_rows))
    script = f"""#!/usr/bin/env bash
set -euo pipefail

source /home/agent/wja/miniconda3/etc/profile.d/conda.sh
conda activate cospaq

cd {REPO_ROOT}

python artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/scripts/benchmark_broad_grid_vllm_parallel.py \\
  --output-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/focused_uniform_vllm \\
  --methods dense_bf16,dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4 \\
  --gpus "${{GPUS:-0,1,2,3,4}}" \\
  --batches {batches} \\
  --input-seqs {input_seqs} \\
  --output-seqs 1 \\
  --warmup-iters 1 \\
  --iters 5 \\
  --continue-on-error

python artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/solve_promising_policies.py \\
  --scenarios artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/summary/focused_retest_scenarios.csv \\
  --output-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/policies

python artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/export_promising_policy_checkpoints.py \\
  --policy-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/policies/unique_policies \\
  --output-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/checkpoints \\
  --force

python artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/benchmark_promising_policy_vllm_parallel.py \\
  --policy-csv artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/policies/scenario_policies.csv \\
  --checkpoint-root artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/checkpoints \\
  --output-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/focused_hetero_vllm \\
  --method-name focused_optimized_hetero \\
  --output-prefix focused_optimized_hetero \\
  --gpus "${{GPUS:-0}}" \\
  --scenarios {scenarios} \\
  --warmup-iters 1 \\
  --iters 5
"""
    path = output_dir / "run_focused_retest.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def parse_ints(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def quality_bin(quality: float, scale: float) -> int:
    if quality <= 0:
        return 0
    return max(1, int(math.ceil(quality * scale)))


def counter_text(counter: Counter[str]) -> str:
    return ",".join(f"{method}:{counter.get(method, 0)}" for method in HETERO_METHODS if counter.get(method, 0))


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
