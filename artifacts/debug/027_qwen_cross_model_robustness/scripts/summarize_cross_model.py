#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = SCRIPT_DIR.parents[0]
MODEL_VARIANTS = ("0.8B", "2B", "4B", "9B", "llama2-7b", "llama31-8b")
MODEL_LABELS = {
    "0.8B": "Qwen3.5-0.8B",
    "2B": "Qwen3.5-2B",
    "4B": "Qwen3.5-4B",
    "9B": "Qwen3.5-9B",
    "llama2-7b": "Llama-2-7B",
    "llama31-8b": "Llama-3.1-8B",
}
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
    parser = argparse.ArgumentParser(description="Summarize Qwen cross-model robustness results.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speed_path = args.output_root / "speed" / "qwen_cross_model_raw.csv"
    rows = read_csv(speed_path)
    if not rows:
        raise RuntimeError(f"missing speed rows: {speed_path}")
    best = best_rows(rows)
    table_rows = build_main_table_rows(best)
    summary_dir = args.output_root / "summary"
    write_csv(summary_dir / "model_workload_method_table.csv", table_rows)
    write_text(summary_dir / "model_workload_method_table.md", render_main_table(table_rows))

    model_rows = build_model_average_rows(table_rows)
    write_csv(summary_dir / "model_average_table.csv", model_rows)
    write_text(summary_dir / "model_average_table.md", render_model_average_table(model_rows))

    transfer_rows = build_transfer_rows(model_rows)
    write_csv(summary_dir / "cross_model_transfer.csv", transfer_rows)
    write_text(summary_dir / "cross_model_transfer.md", render_transfer_table(transfer_rows))
    write_text(summary_dir / "analysis.md", render_analysis(model_rows, transfer_rows))
    print(f"wrote summary to {summary_dir}")


def best_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["model_variant"], row["scenario"], row["method"])
        prev = best.get(key)
        if prev is None or rank(row) > rank(prev):
            best[key] = row
    return best


def rank(row: dict[str, str]) -> tuple[int, str]:
    return (int(float(row.get("iters") or 0)), row.get("timestamp", ""))


def build_main_table_rows(best: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for model in MODEL_VARIANTS:
        for scenario in SCENARIOS:
            dense = best.get((model, scenario, "dense_bf16"))
            dense_ms = f(dense, "e2e_ms") if dense is not None else 0.0
            row: dict[str, Any] = {"model_variant": model, "scenario": scenario}
            for method in METHODS:
                item = best.get((model, scenario, method))
                if item is None or dense_ms <= 0:
                    row[f"{method}_e2e_ms"] = ""
                    row[f"{method}_speedup_vs_dense"] = ""
                    continue
                e2e_ms = f(item, "e2e_ms")
                row[f"{method}_e2e_ms"] = f"{e2e_ms:.6f}"
                row[f"{method}_speedup_vs_dense"] = f"{dense_ms / e2e_ms:.6f}" if e2e_ms > 0 else ""
            rows.append(row)
    return rows


def build_model_average_rows(table_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in MODEL_VARIANTS:
        scenario_rows = [row for row in table_rows if row["model_variant"] == model]
        for average_name, average_fn in (("arith_mean", arithmetic_mean), ("geomean", geometric_mean)):
            row: dict[str, Any] = {"model_variant": model, "average": average_name}
            for method in METHODS:
                speedups = [
                    f(item, f"{method}_speedup_vs_dense")
                    for item in scenario_rows
                    if item.get(f"{method}_speedup_vs_dense")
                ]
                row[f"{method}_speedup_vs_dense"] = f"{average_fn(speedups):.6f}" if speedups else ""
            rows.append(row)
    for average_name, average_fn in (("overall_arith_mean", arithmetic_mean), ("overall_geomean", geometric_mean)):
        row = {"model_variant": "all_models", "average": average_name}
        for method in METHODS:
            speedups = [
                f(item, f"{method}_speedup_vs_dense")
                for item in table_rows
                if item.get(f"{method}_speedup_vs_dense")
            ]
            row[f"{method}_speedup_vs_dense"] = f"{average_fn(speedups):.6f}" if speedups else ""
        rows.append(row)
    return rows


def build_transfer_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    geomean_rows = [row for row in model_rows if row["average"] == "geomean"]
    transfer_methods = []
    for model in MODEL_VARIANTS:
        row = next((item for item in geomean_rows if item["model_variant"] == model), None)
        if row is None:
            continue
        candidates = [
            (method, f(row, f"{method}_speedup_vs_dense"))
            for method in UNIFORM_METHODS
            if row.get(f"{method}_speedup_vs_dense")
        ]
        if candidates:
            best_uniform = max(candidates, key=lambda item: item[1])[0]
            transfer_methods.append((f"{model}_best_uniform", best_uniform))
    transfer_methods.append(("our_linear_hybrid", "our_linear_hybrid"))

    rows = []
    for label, method in transfer_methods:
        row: dict[str, Any] = {"strategy": label, "method_used": method}
        speedups = []
        for model in MODEL_VARIANTS:
            source = next((item for item in geomean_rows if item["model_variant"] == model), None)
            speedup = source.get(f"{method}_speedup_vs_dense", "") if source else ""
            row[f"{model}_geomean_speedup"] = speedup
            if speedup:
                speedups.append(float(speedup))
        row["arith_mean_speedup"] = f"{arithmetic_mean(speedups):.6f}" if speedups else ""
        row["geomean_speedup"] = f"{geometric_mean(speedups):.6f}" if speedups else ""
        rows.append(row)
    return rows


def render_main_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen/Llama Cross-Model Method Table",
        "",
        "Cells show `speedup_vs_dense_bf16 (e2e_ms)`.",
        "",
        "| Model | Scenario | " + " | ".join(METHODS) + " |",
        "|---|---|" + "|".join("---:" for _ in METHODS) + "|",
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
        lines.append(f"| `{model_label(row['model_variant'])}` | `{row['scenario']}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_model_average_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen/Llama Cross-Model Average Table",
        "",
        "| Model | Average | " + " | ".join(METHODS) + " |",
        "|---|---|" + "|".join("---:" for _ in METHODS) + "|",
    ]
    for row in rows:
        cells = []
        for method in METHODS:
            speedup = row.get(f"{method}_speedup_vs_dense", "")
            cells.append(f"{float(speedup):.3f}x" if speedup else "")
        model = "all_models" if row["model_variant"] == "all_models" else model_label(row["model_variant"])
        lines.append(f"| `{model}` | `{row['average']}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_transfer_table(rows: list[dict[str, Any]]) -> str:
    fields = ["strategy", "method_used", *[f"{model}_geomean_speedup" for model in MODEL_VARIANTS], "arith_mean_speedup", "geomean_speedup"]
    lines = [
        "# Qwen/Llama Cross-Model Transfer Table",
        "",
        "| " + " | ".join(fields) + " |",
        "|---|" + "|".join("---:" if "speedup" in field else "---" for field in fields[1:]) + "|",
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


def render_analysis(model_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> str:
    overall = next((row for row in model_rows if row["model_variant"] == "all_models" and row["average"] == "overall_geomean"), None)
    geomean = {row["strategy"]: f(row, "geomean_speedup") for row in transfer_rows if row.get("geomean_speedup")}
    best = max(geomean.items(), key=lambda item: item[1]) if geomean else ("", 0.0)
    lines = [
        "# Qwen/Llama Cross-Model Robustness Analysis",
        "",
        f"- Models: `{', '.join(model_label(model) for model in MODEL_VARIANTS)}`.",
        f"- Workloads: `{', '.join(SCENARIOS)}`.",
        "- Main table: `summary/model_workload_method_table.md`.",
        "- Model average table: `summary/model_average_table.md`.",
        "- Transfer table: `summary/cross_model_transfer.md`.",
    ]
    if overall is not None:
        ours = overall.get("our_linear_hybrid_speedup_vs_dense", "")
        if ours:
            lines.append(f"- `our_linear_hybrid` overall geomean: `{float(ours):.3f}x`.")
    if best[0]:
        lines.append(f"- Best transfer strategy in current results: `{best[0]}` at `{best[1]:.3f}x`.")
    lines.extend(
        [
            "",
            "Interpretation target:",
            "- Uniform methods should show model-size-dependent winners and weaker transferred averages.",
            "- `our_linear_hybrid` should retain strong average speedup by selecting per-linear prefill/decode backends per model and workload.",
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


def f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
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


def model_label(variant: str) -> str:
    return MODEL_LABELS.get(variant, variant)


if __name__ == "__main__":
    main()
