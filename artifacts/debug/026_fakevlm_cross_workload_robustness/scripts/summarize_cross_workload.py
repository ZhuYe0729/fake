#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = SCRIPT_DIR.parents[0]
SCENARIOS = ("prefill_only", "normal_01", "normal_02")
METHODS = (
    "dense_bf16",
    "uniform_dense_nvfp4",
    "uniform_sparse_bf16",
    "uniform_sparse_nvfp4",
    "uniform_marlin_weight_only",
    "uniform_dense_nvfp4_prefill_marlin_decode",
    "our_linear_hybrid",
)
UNIFORM_METHODS = tuple(method for method in METHODS if method.startswith("uniform_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FakeVLM cross-workload robustness results.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speed_path = args.output_root / "speed" / "e2e_speed_raw.csv"
    rows = read_csv(speed_path)
    if not rows:
        raise RuntimeError(f"missing speed rows: {speed_path}")
    best = best_rows(rows)
    table_rows = build_table_rows(best)
    summary_dir = args.output_root / "summary"
    write_csv(summary_dir / "workload_method_table.csv", table_rows)
    write_text(summary_dir / "workload_method_table.md", render_main_table(table_rows))

    transfer_rows = build_transfer_rows(best)
    write_csv(summary_dir / "cross_workload_transfer.csv", transfer_rows)
    write_text(summary_dir / "cross_workload_transfer.md", render_transfer_table(transfer_rows))
    write_text(summary_dir / "analysis.md", render_analysis(table_rows, transfer_rows))
    print(f"wrote summary to {summary_dir}")


def best_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["scenario"], row["method"])
        prev = best.get(key)
        if prev is None or rank(row) > rank(prev):
            best[key] = row
    return best


def rank(row: dict[str, str]) -> tuple[int, str]:
    return (int(float(row.get("iters") or 0)), row.get("timestamp", ""))


def build_table_rows(best: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    dense_by_scenario = {}
    for scenario in SCENARIOS:
        dense = best.get((scenario, "dense_bf16"))
        if dense is None:
            continue
        dense_by_scenario[scenario] = f(dense, "e2e_ms")

    rows = []
    for scenario in SCENARIOS:
        dense_ms = dense_by_scenario.get(scenario)
        row: dict[str, Any] = {"scenario": scenario}
        for method in METHODS:
            item = best.get((scenario, method))
            if item is None or dense_ms is None:
                row[f"{method}_e2e_ms"] = ""
                row[f"{method}_speedup_vs_dense"] = ""
                continue
            e2e_ms = f(item, "e2e_ms")
            row[f"{method}_e2e_ms"] = f"{e2e_ms:.6f}"
            row[f"{method}_speedup_vs_dense"] = f"{dense_ms / e2e_ms:.6f}" if e2e_ms > 0 else ""
        rows.append(row)

    for average_name, average_fn in (("arith_mean", arithmetic_mean), ("geomean", geometric_mean)):
        row = {"scenario": average_name}
        for method in METHODS:
            speedups = [f(item, f"{method}_speedup_vs_dense") for item in rows if item["scenario"] in SCENARIOS and item.get(f"{method}_speedup_vs_dense")]
            row[f"{method}_e2e_ms"] = ""
            row[f"{method}_speedup_vs_dense"] = f"{average_fn(speedups):.6f}" if speedups else ""
        rows.append(row)
    return rows


def build_transfer_rows(best: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    table_rows = build_table_rows(best)
    scenario_rows = [row for row in table_rows if row["scenario"] in SCENARIOS]
    transfer_methods = []
    for source in SCENARIOS:
        source_row = next((row for row in scenario_rows if row["scenario"] == source), None)
        if source_row is None:
            continue
        candidates = [
            (method, f(source_row, f"{method}_speedup_vs_dense"))
            for method in UNIFORM_METHODS
            if source_row.get(f"{method}_speedup_vs_dense")
        ]
        if not candidates:
            continue
        best_uniform = max(candidates, key=lambda item: item[1])[0]
        transfer_methods.append((f"{source}_best_uniform", best_uniform))
    transfer_methods.append(("our_linear_hybrid", "our_linear_hybrid"))

    rows = []
    for label, method in transfer_methods:
        row: dict[str, Any] = {
            "strategy": label,
            "method_used": method,
        }
        speedups = []
        for scenario_row in scenario_rows:
            speedup = scenario_row.get(f"{method}_speedup_vs_dense", "")
            row[f"{scenario_row['scenario']}_speedup_vs_dense"] = speedup
            if speedup:
                speedups.append(float(speedup))
        row["arith_mean_speedup"] = f"{arithmetic_mean(speedups):.6f}" if speedups else ""
        row["geomean_speedup"] = f"{geometric_mean(speedups):.6f}" if speedups else ""
        rows.append(row)
    return rows


def render_main_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FakeVLM Cross-Workload Method Table",
        "",
        "Cells show `speedup_vs_dense_bf16 (e2e_ms)`.",
        "",
        "| Scenario | " + " | ".join(METHODS) + " |",
        "|---|" + "|".join("---:" for _ in METHODS) + "|",
    ]
    for row in rows:
        cells = []
        for method in METHODS:
            speedup = row.get(f"{method}_speedup_vs_dense", "")
            latency = row.get(f"{method}_e2e_ms", "")
            if not speedup:
                cells.append("")
            elif latency:
                cells.append(f"{float(speedup):.3f}x ({float(latency):.1f} ms)")
            else:
                cells.append(f"{float(speedup):.3f}x")
        lines.append(f"| `{row['scenario']}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_transfer_table(rows: list[dict[str, Any]]) -> str:
    fields = ["strategy", "method_used", *[f"{scenario}_speedup_vs_dense" for scenario in SCENARIOS], "arith_mean_speedup", "geomean_speedup"]
    lines = [
        "# FakeVLM Cross-Workload Transfer Table",
        "",
        "| " + " | ".join(fields) + " |",
        "|---|" + "|".join("---:" if field.endswith("speedup") or "speedup_vs_dense" in field else "---" for field in fields[1:]) + "|",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if "speedup" in field and value != "":
                value = f"{float(value):.3f}x"
            elif field in {"strategy", "method_used"}:
                value = f"`{value}`"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def render_analysis(table_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> str:
    geomean = {row["strategy"]: f(row, "geomean_speedup") for row in transfer_rows if row.get("geomean_speedup")}
    best = max(geomean.items(), key=lambda item: item[1]) if geomean else ("", 0.0)
    lines = [
        "# FakeVLM Cross-Workload Robustness Analysis",
        "",
        f"- Workloads: `{', '.join(SCENARIOS)}`.",
        "- Main table: `summary/workload_method_table.md`.",
        "- Transfer table: `summary/cross_workload_transfer.md`.",
    ]
    if best[0]:
        lines.append(f"- Best geomean strategy in current results: `{best[0]}` at `{best[1]:.3f}x`.")
    lines.extend(
        [
            "",
            "Interpretation target:",
            "- Uniform methods should show workload-dependent winners and weaker transferred averages.",
            "- `our_linear_hybrid` should retain strong average speedup by selecting per-linear prefill/decode backends for each workload.",
            "",
        ]
    )
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def arithmetic_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def geometric_mean(values: list[float]) -> float:
    positives = [value for value in values if value > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


if __name__ == "__main__":
    main()
