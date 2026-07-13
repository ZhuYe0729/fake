#!/usr/bin/env python3
"""Summarize promising scenario optimized-policy vLLM retest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RETEST_ROOT = SCRIPT_DIR.parent
BASELINE_ROOT = RETEST_ROOT.parent
REPO_ROOT = BASELINE_ROOT.parents[3]
DEFAULT_POLICY_CSV = RETEST_ROOT / "policies/scenario_policies.csv"
DEFAULT_BROAD_SUMMARY = BASELINE_ROOT / "broad_grid_vllm/results/summary_long.csv"
DEFAULT_OPT_SUMMARY = RETEST_ROOT / "benchmarks/optimized_hetero_vllm/optimized_hetero_summary.csv"
DEFAULT_POLICY_QUALITY = RETEST_ROOT / "quality/optimized_policy_quality.csv"
DEFAULT_SINGLE_QUALITY = REPO_ROOT / "artifacts/debug/018_llama2_prefill_global_pareto/report/final_full_arc_c_report.csv"
DEFAULT_UNIQUE_POLICY_DIR = RETEST_ROOT / "policies/unique_policies"
DEFAULT_MAX_POLICY_CSV = RETEST_ROOT / "max_speed/policies/scenario_policies.csv"
DEFAULT_MAX_SUMMARY = RETEST_ROOT / "max_speed/benchmarks/max_speed_hetero_vllm/max_speed_hetero_summary.csv"
DEFAULT_MAX_POLICY_QUALITY = RETEST_ROOT / "max_speed/quality/optimized_policy_quality.csv"
DEFAULT_MAX_UNIQUE_POLICY_DIR = RETEST_ROOT / "max_speed/policies/unique_policies"
DEFAULT_OUTPUT_DIR = RETEST_ROOT / "summary"
SINGLE_METHODS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-csv", type=Path, default=DEFAULT_POLICY_CSV)
    parser.add_argument("--broad-summary", type=Path, default=DEFAULT_BROAD_SUMMARY)
    parser.add_argument("--optimized-summary", type=Path, default=DEFAULT_OPT_SUMMARY)
    parser.add_argument("--policy-quality", type=Path, default=DEFAULT_POLICY_QUALITY)
    parser.add_argument("--single-quality", type=Path, default=DEFAULT_SINGLE_QUALITY)
    parser.add_argument("--unique-policy-dir", type=Path, default=DEFAULT_UNIQUE_POLICY_DIR)
    parser.add_argument("--max-policy-csv", type=Path, default=DEFAULT_MAX_POLICY_CSV)
    parser.add_argument("--max-summary", type=Path, default=DEFAULT_MAX_SUMMARY)
    parser.add_argument("--max-policy-quality", type=Path, default=DEFAULT_MAX_POLICY_QUALITY)
    parser.add_argument("--max-unique-policy-dir", type=Path, default=DEFAULT_MAX_UNIQUE_POLICY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = read_csv(args.policy_csv)
    broad = read_csv(args.broad_summary)
    optimized = read_csv(args.optimized_summary)
    policy_quality = read_csv_if_exists(args.policy_quality)
    max_policies = read_csv_if_exists(args.max_policy_csv)
    max_summary = read_csv_if_exists(args.max_summary)
    max_policy_quality = read_csv_if_exists(args.max_policy_quality)
    single_quality = read_csv_if_exists(args.single_quality)
    policy_details = read_policy_details("optimized_hetero", args.unique_policy_dir, policies, policy_quality)
    max_policy_details = read_policy_details("max_speed_hetero", args.max_unique_policy_dir, max_policies, max_policy_quality)
    policy_quality_by_name = {row["policy_name"]: row for row in policy_quality}
    max_quality_by_name = {row["policy_name"]: row for row in max_policy_quality}
    single_quality_by_method = single_quality_map(single_quality)
    broad_by_key = {
        (row["method"], int(row["batch"]), int(row["input_seq"]), int(row["output_seq"])): row
        for row in broad
    }
    opt_by_scenario = {row["scenario"]: row for row in optimized}
    max_policy_by_scenario = {row["scenario"]: row for row in max_policies}
    max_by_scenario = {row["scenario"]: row for row in max_summary}

    rows: list[dict[str, Any]] = []
    for policy in policies:
        batch = int(policy["batch"])
        input_seq = int(policy["input_seq"])
        output_seq = int(policy["output_seq"])
        dense = broad_by_key.get(("dense_bf16", batch, input_seq, output_seq))
        original = broad_by_key.get(("hetero", batch, input_seq, output_seq))
        opt = opt_by_scenario.get(policy["scenario"])
        single_candidates = []
        for method in SINGLE_METHODS:
            row = broad_by_key.get((method, batch, input_seq, output_seq))
            latency = latency_ms(row)
            if latency is not None:
                single_candidates.append((latency, method, row))
        best_single_latency, best_single_method, _best_row = min(single_candidates) if single_candidates else (None, "", None)
        dense_latency = latency_ms(dense)
        original_latency = latency_ms(original)
        optimized_latency = latency_ms(opt)
        max_policy = max_policy_by_scenario.get(policy["scenario"], {})
        max_row = max_by_scenario.get(policy["scenario"])
        max_latency = latency_ms(max_row)
        quality = policy_quality_by_name.get(policy["policy_name"], {})
        max_quality = max_quality_by_name.get(max_policy.get("policy_name", ""), {})
        rows.append(
            {
                "scenario": policy["scenario"],
                "batch": batch,
                "input_seq": input_seq,
                "output_seq": output_seq,
                "policy_name": policy["policy_name"],
                "quality_budget": policy["quality_budget"],
                "quality_cost": policy["quality_cost"],
                "policy_predicted_speedup_vs_dense": policy["predicted_speedup_vs_dense"],
                "dense_bf16_ms": fmt(dense_latency),
                "best_single_method": best_single_method,
                "best_single_ms": fmt(best_single_latency),
                "original_hetero_ms": fmt(original_latency),
                "optimized_hetero_ms": fmt(optimized_latency),
                "optimized_status": opt.get("status", "MISSING") if opt else "MISSING",
                "optimized_speedup_vs_dense": ratio(dense_latency, optimized_latency),
                "optimized_speedup_vs_best_single": ratio(best_single_latency, optimized_latency),
                "optimized_speedup_vs_best_single_label": labeled_ratio(best_single_latency, optimized_latency, best_single_method),
                "optimized_speedup_vs_original_hetero": ratio(original_latency, optimized_latency),
                "optimized_arc_acc": quality.get("arc_acc", ""),
                "optimized_arc_acc_norm": quality.get("arc_acc_norm", ""),
                "optimized_quality_sample_len": quality.get("sample_len", ""),
                "max_speed_policy_name": max_policy.get("policy_name", ""),
                "max_speed_quality_cost": max_policy.get("quality_cost", ""),
                "max_speed_hetero_ms": fmt(max_latency),
                "max_speed_status": max_row.get("status", "MISSING") if max_row else "MISSING",
                "max_speed_speedup_vs_dense": ratio(dense_latency, max_latency),
                "max_speed_speedup_vs_best_single": ratio(best_single_latency, max_latency),
                "max_speed_speedup_vs_best_single_label": labeled_ratio(best_single_latency, max_latency, best_single_method),
                "max_speed_arc_acc": max_quality.get("arc_acc", ""),
                "max_speed_arc_acc_norm": max_quality.get("arc_acc_norm", ""),
                "max_speed_quality_sample_len": max_quality.get("sample_len", ""),
                "count_dense_bf16": policy["count_dense_bf16"],
                "count_dense_nvfp4": policy["count_dense_nvfp4"],
                "count_sparse_bf16": policy["count_sparse_bf16"],
                "count_sparse_nvfp4": policy["count_sparse_nvfp4"],
                "measured_best_method_before": policy["measured_best_method"],
                "measured_best_speedup_before": policy["measured_best_speedup"],
                "measured_hetero_speedup_before": policy["measured_hetero_speedup"],
            }
        )

    write_csv(args.output_dir / "promising_policy_retest_summary.csv", rows)
    write_markdown(args.output_dir / "promising_policy_retest_summary.md", rows)
    write_speedup_wide_tables(
        args.output_dir,
        rows,
        broad_by_key,
        opt_by_scenario,
        single_quality_by_method,
        policy_details + max_policy_details,
    )
    write_csv(args.output_dir / "promising_policy_details.csv", policy_details + max_policy_details)
    write_json(
        args.output_dir / "promising_policy_retest_summary_metadata.json",
        {
            "rows": len(rows),
            "single_methods": SINGLE_METHODS,
            "broad_summary": str(args.broad_summary),
            "optimized_summary": str(args.optimized_summary),
            "policy_quality": str(args.policy_quality),
            "max_policy_csv": str(args.max_policy_csv),
            "max_summary": str(args.max_summary),
            "max_policy_quality": str(args.max_policy_quality),
            "single_quality": str(args.single_quality),
        },
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def latency_ms(row: dict[str, str] | None) -> float | None:
    if not row or row.get("status") != "OK":
        return None
    try:
        return float(row["median_ms"])
    except (KeyError, TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def ratio(base: float | None, value: float | None) -> str:
    if base is None or value in (None, 0):
        return ""
    return f"{base / value:.3f}"


def labeled_ratio(base: float | None, value: float | None, label: str) -> str:
    value_str = ratio(base, value)
    return "" if not value_str else f"{value_str} ({label})"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    avg_vs_dense = mean_float(row["optimized_speedup_vs_dense"] for row in rows)
    avg_vs_best = mean_float(row["optimized_speedup_vs_best_single"] for row in rows)
    avg_vs_original = mean_float(row["optimized_speedup_vs_original_hetero"] for row in rows)
    wins_vs_best = [row for row in rows if to_float(row["optimized_speedup_vs_best_single"]) > 1.0]
    lines = [
        "# Promising Scenario Optimized Policy Retest",
        "",
        "本表复用 broad-grid 已测 single 方法和原 `hetero` 结果，只新增测试 P024 精度预算下重新求解的 `optimized_hetero` vLLM checkpoint。",
        "",
        "## Aggregate",
        "",
        f"- Mean `optimized_hetero` speedup vs dense bf16: {avg_vs_dense:.3f}x.",
        f"- Mean `optimized_hetero` speedup vs best single: {avg_vs_best:.3f}x ({len(wins_vs_best)}/{len(rows)} scenarios > 1).",
        f"- Mean `optimized_hetero` speedup vs original hetero: {avg_vs_original:.3f}x.",
        "- Scenarios faster than best single: "
        + (
            ", ".join(
                f"{row['scenario']} ({row['optimized_speedup_vs_best_single']}x)"
                for row in wins_vs_best
            )
            if wins_vs_best
            else "none"
        )
        + ".",
        "",
        "## Summary Table",
        "",
        "| scenario | best_single | original_hetero_ms | optimized_hetero_ms | opt_vs_dense | opt_vs_best_single | opt_vs_original_hetero | policy | quality_cost | method_counts |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        counts = (
            f"dense={row['count_dense_bf16']}, "
            f"dnvfp4={row['count_dense_nvfp4']}, "
            f"sbf16={row['count_sparse_bf16']}, "
            f"snvfp4={row['count_sparse_nvfp4']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    f"{row['best_single_method']}:{row['best_single_ms']}",
                    str(row["original_hetero_ms"]),
                    str(row["optimized_hetero_ms"]),
                    str(row["optimized_speedup_vs_dense"]),
                    str(row["optimized_speedup_vs_best_single"]),
                    str(row["optimized_speedup_vs_original_hetero"]),
                    str(row["policy_name"]),
                    str(row["quality_cost"]),
                    counts,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `optimized_hetero` policies use the 018 P024 quality budget and exclude Marlin from the optimizer because no trusted Marlin quality proxy exists in that run.",
            "- Single-method latency and original hetero latency are copied from `broad_grid_vllm/results/summary_long.csv`.",
            "- `opt_vs_best_single > 1` means the new quality-constrained mixed policy is faster than the best already-tested single method for that scenario.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_speedup_wide_tables(
    output_dir: Path,
    rows: list[dict[str, Any]],
    broad_by_key: dict[tuple[str, int, int, int], dict[str, str]],
    opt_by_scenario: dict[str, dict[str, str]],
    single_quality_by_method: dict[str, dict[str, str]],
    policy_details: list[dict[str, Any]],
) -> None:
    fields = [
        "scenario",
        "batch",
        "input_seq",
        "output_seq",
        "dense_bf16_speedup",
        "dense_nvfp4_speedup",
        "sparse_bf16_speedup",
        "sparse_nvfp4_speedup",
        "marlin_nvfp4_speedup",
        "original_hetero_speedup",
        "optimized_hetero_speedup",
        "optimized_vs_best_single_speedup",
        "optimized_arc_acc_norm",
        "optimized_policy",
        "optimized_quality_cost",
        "max_speed_hetero_speedup",
        "max_speed_vs_best_single_speedup",
        "max_speed_arc_acc_norm",
        "max_speed_policy",
        "max_speed_quality_cost",
    ]
    wide_rows = []
    for row in rows:
        batch = int(row["batch"])
        input_seq = int(row["input_seq"])
        output_seq = int(row["output_seq"])
        dense_latency = latency_ms(broad_by_key.get(("dense_bf16", batch, input_seq, output_seq)))
        wide_row = {
            "scenario": row["scenario"],
            "batch": batch,
            "input_seq": input_seq,
            "output_seq": output_seq,
            "dense_bf16_speedup": ratio(dense_latency, dense_latency),
            "optimized_policy": row["policy_name"],
            "optimized_quality_cost": row["quality_cost"],
        }
        for method in SINGLE_METHODS[1:]:
            method_latency = latency_ms(broad_by_key.get((method, batch, input_seq, output_seq)))
            wide_row[f"{method}_speedup"] = ratio(dense_latency, method_latency)
        original_latency = latency_ms(broad_by_key.get(("hetero", batch, input_seq, output_seq)))
        optimized_latency = latency_ms(opt_by_scenario.get(str(row["scenario"])))
        wide_row["original_hetero_speedup"] = ratio(dense_latency, original_latency)
        wide_row["optimized_hetero_speedup"] = ratio(dense_latency, optimized_latency)
        wide_row["optimized_vs_best_single_speedup"] = row["optimized_speedup_vs_best_single_label"]
        wide_row["optimized_arc_acc_norm"] = fmt_float(row["optimized_arc_acc_norm"])
        wide_row["max_speed_hetero_speedup"] = row["max_speed_speedup_vs_dense"]
        wide_row["max_speed_vs_best_single_speedup"] = row["max_speed_speedup_vs_best_single_label"]
        wide_row["max_speed_arc_acc_norm"] = fmt_float(row["max_speed_arc_acc_norm"])
        wide_row["max_speed_policy"] = row["max_speed_policy_name"]
        wide_row["max_speed_quality_cost"] = row["max_speed_quality_cost"]
        wide_rows.append(wide_row)

    write_csv(output_dir / "promising_policy_retest_speedup_wide.csv", wide_rows)
    lines = [
        "# Promising Scenario Speedup Wide Table",
        "",
        "All speedups are measured vLLM median-latency speedups over dense bf16 for the same `(batch, input_seq, output_seq)` scenario. `*_acc_norm` columns are full ARC-Challenge 0-shot `acc_norm` measured with vLLM + lm-eval.",
        "",
        "| batch | input_seq | output_seq | dense_bf16 | dense_nvfp4 | sparse_bf16 | sparse_nvfp4 | marlin_nvfp4 | original_hetero | optimized_hetero | opt_vs_best_single | optimized_acc_norm | optimized_policy | max_speed_hetero | max_speed_vs_best_single | max_speed_acc_norm | max_speed_policy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in wide_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["batch"]),
                    str(row["input_seq"]),
                    str(row["output_seq"]),
                    str(row["dense_bf16_speedup"]),
                    str(row["dense_nvfp4_speedup"]),
                    str(row["sparse_bf16_speedup"]),
                    str(row["sparse_nvfp4_speedup"]),
                    str(row["marlin_nvfp4_speedup"]),
                    str(row["original_hetero_speedup"]),
                    str(row["optimized_hetero_speedup"]),
                    str(row["optimized_vs_best_single_speedup"]),
                    str(row["optimized_arc_acc_norm"]),
                    str(row["optimized_policy"]),
                    str(row["max_speed_hetero_speedup"]),
                    str(row["max_speed_vs_best_single_speedup"]),
                    str(row["max_speed_arc_acc_norm"]),
                    str(row["max_speed_policy"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `original_hetero` is the previously measured broad-grid hetero baseline.",
            "- `optimized_hetero` is the newly exported and measured P024 quality-budget layer-wise heterogeneous policy for each scenario.",
            "- `opt_vs_best_single` is `optimized_hetero` latency speedup over the fastest measured single method in that scenario; the method is shown in parentheses.",
            "- `max_speed_hetero` is the newly exported and measured unconstrained speed-optimal heterogeneous policy for each scenario.",
            "",
            "## Single-Method Quality",
            "",
            "| method | ARC-C acc | ARC-C acc_norm | NLL | sample_len | source |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for method in SINGLE_METHODS:
        quality = single_quality_by_method.get(method, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt_float(quality.get("arc_acc", "")),
                    fmt_float(quality.get("arc_acc_norm", "")),
                    fmt_float(quality.get("nll", "")),
                    str(quality.get("arc_sample_len", "")),
                    str(quality.get("source", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Optimized Policy Details",
            "",
            "| kind | policy | ARC-C acc_norm | quality_cost | method_counts | scenarios | assignment_summary |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for detail in policy_details:
        counts = (
            f"dense={detail['count_dense_bf16']}, "
            f"dnvfp4={detail['count_dense_nvfp4']}, "
            f"sbf16={detail['count_sparse_bf16']}, "
            f"snvfp4={detail['count_sparse_nvfp4']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(detail["policy_kind"]),
                    str(detail["policy_name"]),
                    fmt_float(detail.get("arc_acc_norm", "")),
                    str(detail["quality_cost"]),
                    counts,
                    str(detail["scenarios"]),
                    markdown_cell(str(detail["assignment_summary"])),
                ]
            )
            + " |"
        )
    (output_dir / "promising_policy_retest_speedup_wide.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def single_quality_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        if row.get("row_type") != "uniform":
            continue
        label = row.get("label", "")
        if not label.startswith("all_"):
            continue
        result[label.removeprefix("all_")] = row
    return result


def read_policy_details(
    policy_kind: str,
    unique_policy_dir: Path,
    scenario_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    scenarios_by_policy: dict[str, list[str]] = {}
    summary_by_policy: dict[str, dict[str, str]] = {}
    for row in scenario_rows:
        policy_name = row["policy_name"]
        scenarios_by_policy.setdefault(policy_name, []).append(row["scenario"])
        summary_by_policy.setdefault(policy_name, row)
    quality_by_policy = {row["policy_name"]: row for row in quality_rows}
    details = []
    for path in sorted(unique_policy_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy_name = str(payload["policy_name"])
        summary = summary_by_policy.get(policy_name, payload.get("summary", {}))
        quality = quality_by_policy.get(policy_name, {})
        details.append(
            {
                "policy_kind": policy_kind,
                "policy_name": policy_name,
                "quality_cost": summary.get("quality_cost", ""),
                "arc_acc": quality.get("arc_acc", ""),
                "arc_acc_norm": quality.get("arc_acc_norm", ""),
                "sample_len": quality.get("sample_len", ""),
                "count_dense_bf16": summary.get("count_dense_bf16", ""),
                "count_dense_nvfp4": summary.get("count_dense_nvfp4", ""),
                "count_sparse_bf16": summary.get("count_sparse_bf16", ""),
                "count_sparse_nvfp4": summary.get("count_sparse_nvfp4", ""),
                "scenarios": ", ".join(scenarios_by_policy.get(policy_name, [])),
                "assignment_summary": summarize_assignments(payload.get("assignments", {})),
            }
        )
    return details


def summarize_assignments(assignments: dict[str, str]) -> str:
    grouped: dict[str, dict[str, list[int]]] = {}
    for module, method in assignments.items():
        parts = module.split(".")
        if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
            continue
        layer = int(parts[2])
        fused_name = ".".join(parts[3:])
        grouped.setdefault(method, {}).setdefault(fused_name, []).append(layer)
    chunks = []
    for method in sorted(grouped):
        group_chunks = []
        for fused_name in sorted(grouped[method]):
            group_chunks.append(f"{fused_name}@{compress_ranges(sorted(grouped[method][fused_name]))}")
        chunks.append(f"{method}: " + "; ".join(group_chunks))
    return " | ".join(chunks)


def compress_ranges(values: list[int]) -> str:
    if not values:
        return ""
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(format_range(start, prev))
        start = prev = value
    ranges.append(format_range(start, prev))
    return ",".join(ranges)


def format_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def fmt_float(value: Any, digits: int = 4) -> str:
    parsed = to_float(value)
    if parsed != parsed:
        return ""
    return f"{parsed:.{digits}f}"


def markdown_cell(value: str) -> str:
    return value.replace("|", "<br>")


def mean_float(values: Any) -> float:
    parsed = [to_float(value) for value in values]
    parsed = [value for value in parsed if value == value]
    return sum(parsed) / len(parsed) if parsed else float("nan")


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
