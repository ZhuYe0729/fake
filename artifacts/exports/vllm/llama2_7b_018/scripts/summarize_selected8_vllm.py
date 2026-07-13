#!/usr/bin/env python3
"""Summarize selected-8 vLLM speed and quality results."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BASELINE_ROOT.parents[3]
SCENARIOS = [
    "single_long_prefill_short_decode",
    "small_batch_long_prefill",
    "b4_medium_prefill",
    "b4_long_prefill",
    "b8_mixed_long_prefill",
    "b16_mixed",
    "b32_throughput_mixed",
    "b64_high_batch",
]
SCENARIO_CONFIGS = {
    "single_long_prefill_short_decode": {"batch": 1, "input_len": 8192, "output_tokens": 16},
    "small_batch_long_prefill": {"batch": 2, "input_len": 4096, "output_tokens": 16},
    "b4_medium_prefill": {"batch": 4, "input_len": 2048, "output_tokens": 16},
    "b4_long_prefill": {"batch": 4, "input_len": 4096, "output_tokens": 32},
    "b8_mixed_long_prefill": {"batch": 8, "input_len": 2048, "output_tokens": 64},
    "b16_mixed": {"batch": 16, "input_len": 1024, "output_tokens": 64},
    "b32_throughput_mixed": {"batch": 32, "input_len": 512, "output_tokens": 64},
    "b64_high_batch": {"batch": 64, "input_len": 256, "output_tokens": 128},
}
METHODS = [
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "hetero",
]
HETERO_SCENARIO_STRATEGY = {
    "single_long_prefill_short_decode": "hetero_strategy_a",
    "small_batch_long_prefill": "hetero_strategy_a",
    "b4_medium_prefill": "hetero_strategy_b",
    "b4_long_prefill": "hetero_strategy_b",
    "b8_mixed_long_prefill": "hetero_strategy_c",
    "b16_mixed": "hetero_strategy_c",
    "b32_throughput_mixed": "hetero_strategy_c",
    "b64_high_batch": "hetero_strategy_c",
}
HETERO_ASSIGNMENTS = {
    "hetero_strategy_a": "qkv/gate_up=dense_nvfp4, o/down=marlin_nvfp4",
    "hetero_strategy_b": "qkv/gate_up=dense_nvfp4, o=dense_bf16, down=marlin_nvfp4",
    "hetero_strategy_c": "qkv/gate_up=sparse_bf16, o=dense_bf16, down=marlin_nvfp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speed-summary",
        type=Path,
        default=BASELINE_ROOT
        / "benchmarks/selected_8_scenarios_vllm/selected8_vllm_summary.csv",
    )
    parser.add_argument(
        "--hetero-quality",
        type=Path,
        default=BASELINE_ROOT / "quality/selected_8_scenarios/selected8_vllm_quality.csv",
    )
    parser.add_argument(
        "--uniform-quality",
        type=Path,
        default=REPO_ROOT
        / "artifacts/debug/018_llama2_prefill_global_pareto/validation/uniform_quality_full_arc_c.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT / "summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    speed_rows = read_csv(args.speed_summary)
    speed_table = build_speed_table(speed_rows)
    quality = load_quality(args.uniform_quality, args.hetero_quality)
    table_rows = build_main_rows(speed_table, quality)
    write_csv(args.output_dir / "selected_8_scenarios_speed_quality.csv", table_rows)
    write_markdown(
        args.output_dir / "selected_8_scenarios_speed_quality.md",
        table_rows,
        speed_rows,
        quality,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_speed_table(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    table = {}
    for row in rows:
        try:
            table[(row["method"], row["scenario"])] = float(row["speedup_vs_dense_bf16"])
        except (KeyError, TypeError, ValueError):
            continue
    return table


def load_quality(uniform_path: Path, hetero_path: Path) -> dict[str, dict[str, Any]]:
    quality: dict[str, dict[str, Any]] = {}
    for row in read_csv(uniform_path):
        method = method_from_uniform_row(row)
        if not method:
            continue
        quality[method] = {
            "arc_acc_norm": float(row["arc_acc_norm"]),
            "arc_acc": float(row["arc_acc"]),
            "nll": float(row["nll"]),
            "nll_delta_vs_dense": float(row["nll"]) - dense_nll(uniform_path),
            "arc_sample_len": int(float(row.get("arc_sample_len") or 1172)),
            "source": "018_uniform_full_arc_c",
        }
    for row in read_csv(hetero_path):
        method = row["method"]
        acc_norm = parse_optional_float(row.get("acc_norm"))
        quality[method] = {
            "arc_acc_norm": acc_norm,
            "arc_acc": parse_optional_float(row.get("acc")),
            "nll": math.nan,
            "nll_delta_vs_dense": math.nan,
            "arc_sample_len": "",
            "source": "selected8_vllm_lm_eval",
        }
    return quality


def dense_nll(uniform_path: Path) -> float:
    for row in read_csv(uniform_path):
        if method_from_uniform_row(row) == "dense_bf16":
            return float(row["nll"])
    return math.nan


def method_from_uniform_row(row: dict[str, str]) -> str | None:
    if "label" in row:
        label = row["label"]
        if label.startswith("all_"):
            return label.removeprefix("all_")
    counts = row.get("backend_counts")
    if counts:
        parsed = ast.literal_eval(counts)
        if len(parsed) == 1:
            return next(iter(parsed))
    return None


def parse_optional_float(value: str | None) -> float:
    if value in (None, "", "None"):
        return math.nan
    return float(value)


def build_main_rows(
    speed_table: dict[tuple[str, str], float],
    quality: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        row: dict[str, Any] = {"method": method}
        speedups = []
        for scenario in SCENARIOS:
            value = speed_table.get((method, scenario), math.nan)
            row[scenario] = value
            if not math.isnan(value):
                speedups.append(value)
        row["avg_speedup"] = sum(speedups) / len(speedups) if speedups else math.nan
        if method == "hetero":
            values = []
            for scenario in SCENARIOS:
                strategy = HETERO_SCENARIO_STRATEGY[scenario]
                q = quality.get(strategy, {})
                value = q.get("arc_acc_norm", math.nan)
                if not math.isnan(value):
                    values.append(value)
            row["arc_c_acc_norm"] = sum(values) / len(values) if values else math.nan
        else:
            row["arc_c_acc_norm"] = quality.get(method, {}).get("arc_acc_norm", math.nan)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["method", *SCENARIOS, "avg_speedup", "arc_c_acc_norm"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    table_rows: list[dict[str, Any]],
    speed_rows: list[dict[str, str]],
    quality: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Llama2-7B selected 8 scenarios vLLM speed and quality",
        "",
        "Speedup is median latency speedup versus `dense_bf16` in the same scenario.",
        "Quality column is full ARC-Challenge `acc_norm` when available.",
        "",
    ]
    headers = ["method", *SCENARIOS, "avg_speedup", "arc_c_acc_norm"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in table_rows:
        cells = [str(row["method"])]
        cells.extend(format_speed(row[scenario]) for scenario in SCENARIOS)
        cells.append(format_speed(row["avg_speedup"]))
        cells.append(format_quality(row["arc_c_acc_norm"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Hetero strategy quality by scenario", ""])
    lines.append("| scenario | strategy | assignment | ARC-C acc_norm | source |")
    lines.append("|---|---|---|---:|---|")
    for scenario in SCENARIOS:
        strategy = HETERO_SCENARIO_STRATEGY[scenario]
        q = quality.get(strategy, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario,
                    strategy,
                    HETERO_ASSIGNMENTS[strategy],
                    format_quality(q.get("arc_acc_norm", math.nan)),
                    str(q.get("source", "pending")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Scenario configs", ""])
    lines.append("| scenario | batch | input_len | output_tokens | prefill_M | hetero_strategy |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for scenario in SCENARIOS:
        config = SCENARIO_CONFIGS[scenario]
        prefill_m = config["batch"] * config["input_len"]
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario,
                    str(config["batch"]),
                    str(config["input_len"]),
                    str(config["output_tokens"]),
                    str(prefill_m),
                    HETERO_SCENARIO_STRATEGY[scenario],
                ]
            )
            + " |"
        )

    lines.extend(["", "## Notes", ""])
    lines.append(
        "- `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`, and "
        "`marlin_nvfp4` quality are loaded from the existing full ARC-C uniform results."
    )
    lines.append(
        "- Hetero quality is loaded from `quality/selected_8_scenarios/selected8_vllm_quality.csv`; "
        "missing values are left as `pending`."
    )
    lines.append(f"- Speed rows loaded: {len(speed_rows)}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_speed(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "pending"
    if math.isnan(value):
        return "pending"
    return f"{value:.3f}x"


def format_quality(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "pending"
    if math.isnan(value):
        return "pending"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
