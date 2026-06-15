#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from common_pareto import DEBUG_ROOT, f, read_csv, write_csv


POINT_NOTES = {
    "point_000": "Dense BF16 reference.",
    "point_013": "Low-loss mixed point: small speedup with nearly dense ARC-C.",
    "point_019": "Intermediate mixed point: shows the smooth low-loss frontier before P020.",
    "point_020": "Conservative mixed point: better NLL and ARC-C than all-dense NVFP4.",
    "point_024": "Main mixed point: faster than uniform sparse baselines with much better quality.",
    "point_026": "Aggressive mixed point: highest recommended speedup before large ARC-C drop.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final report-ready full ARC-Challenge plots.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args.output_root)
    write_csv(report_dir / "final_full_arc_c_report.csv", rows)
    plot_speed_vs_arc(rows, report_dir / "pareto_speed_vs_full_arc_c.png", args.dpi)
    plot_speed_vs_nll(rows, report_dir / "pareto_speed_vs_nll_full_arc_c.png", args.dpi)
    plot_policy_composition(rows, report_dir / "policy_composition_selected.png", args.dpi)
    write_summary(report_dir / "final_report_summary.md", rows)
    print(f"wrote final report outputs to {report_dir}")


def build_rows(output_root: Path) -> list[dict[str, Any]]:
    full = read_csv(output_root / "summary" / "full_arc_c_comparison.csv")
    speed = {row["label"]: row for row in read_csv(output_root / "summary" / "prefill_only_comparison.csv")}
    keep_labels = {
        "point_000",
        "point_013",
        "point_019",
        "point_020",
        "point_024",
        "point_026",
        "all_dense_bf16",
        "all_dense_nvfp4",
        "all_sparse_bf16",
        "all_sparse_nvfp4",
        "all_marlin_nvfp4",
    }
    out = []
    for row in full:
        if row["label"] not in keep_labels:
            continue
        speed_row = speed.get(row["label"], {})
        item = dict(row)
        item["e2e_speedup_vs_dense"] = speed_row.get("e2e_speedup_vs_dense", "")
        item["e2e_prefill_mean_ms"] = speed_row.get("e2e_prefill_mean_ms", "")
        item["report_note"] = POINT_NOTES.get(row["label"], "Uniform baseline." if row["row_type"] == "uniform" else "")
        out.append(item)
    out.sort(key=sort_key)
    return out


def sort_key(row: dict[str, Any]) -> tuple[int, float]:
    if row["row_type"] == "pareto":
        return (0, f(row, "point_index"))
    return (1, f(row, "e2e_speedup_vs_dense"))


def plot_speed_vs_arc(rows: list[dict[str, Any]], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    pareto = [row for row in rows if row["row_type"] == "pareto"]
    uniform = [row for row in rows if row["row_type"] == "uniform"]
    ax.plot(
        [f(row, "e2e_speedup_vs_dense") for row in pareto],
        [f(row, "arc_acc_norm") for row in pareto],
        "o-",
        color="#1f2937",
        linewidth=2.4,
        markersize=7,
        label="Mixed Pareto policies",
        zorder=3,
    )
    for row in uniform:
        ax.scatter(f(row, "e2e_speedup_vs_dense"), f(row, "arc_acc_norm"), marker="s", s=82, color="#dc2626", zorder=4)
        ax.annotate(label(row), (f(row, "e2e_speedup_vs_dense"), f(row, "arc_acc_norm")), xytext=(6, -11), textcoords="offset points", fontsize=8, color="#991b1b")
    ax.set_xlabel("E2E prefill speedup vs dense BF16")
    ax.set_ylabel("ARC-Challenge acc_norm (full, 1172 examples)")
    ax.set_title("Llama2-7B prefill: Speed vs full ARC-Challenge")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_speed_vs_nll(rows: list[dict[str, Any]], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    pareto = [row for row in rows if row["row_type"] == "pareto"]
    uniform = [row for row in rows if row["row_type"] == "uniform"]
    ax.plot(
        [f(row, "e2e_speedup_vs_dense") for row in pareto],
        [f(row, "nll_delta_vs_dense") for row in pareto],
        "o-",
        color="#1f2937",
        linewidth=2.4,
        markersize=7,
        label="Mixed Pareto policies",
        zorder=3,
    )
    for row in uniform:
        ax.scatter(f(row, "e2e_speedup_vs_dense"), f(row, "nll_delta_vs_dense"), marker="s", s=82, color="#dc2626", zorder=4)
        ax.annotate(label(row), (f(row, "e2e_speedup_vs_dense"), f(row, "nll_delta_vs_dense")), xytext=(6, -11), textcoords="offset points", fontsize=8, color="#991b1b")
    ax.set_xlabel("E2E prefill speedup vs dense BF16")
    ax.set_ylabel("NLL delta vs dense BF16")
    ax.set_title("Llama2-7B prefill: Speed vs NLL")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_policy_composition(rows: list[dict[str, Any]], path: Path, dpi: int) -> None:
    pareto = [row for row in rows if row["row_type"] == "pareto"]
    methods = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
    colors = {
        "dense_bf16": "#4c72b0",
        "dense_nvfp4": "#55a868",
        "sparse_bf16": "#c44e52",
        "sparse_nvfp4": "#8172b2",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    x = list(range(len(pareto)))
    bottom = [0] * len(pareto)
    for method in methods:
        vals = [parse_counts(row["backend_counts"]).get(method, 0) for row in pareto]
        ax.bar(x, vals, bottom=bottom, label=method, color=colors[method], edgecolor="white", linewidth=0.5)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels([label(row) for row in pareto])
    ax.set_ylabel("Module count")
    ax.set_title("Selected mixed-policy composition")
    ax.set_ylim(0, 240)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Final Full ARC-Challenge Report",
        "",
        "Use these files for presentation; all ARC-Challenge numbers are full-set results with 1172 examples.",
        "",
        "## Files",
        "",
        "- `pareto_speed_vs_full_arc_c.png`",
        "- `pareto_speed_vs_nll_full_arc_c.png`",
        "- `policy_composition_selected.png`",
        "- `final_full_arc_c_report.csv`",
        "",
        "## Main Table",
        "",
        "| row | speedup | NLL delta | ARC-C acc_norm | note |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {f(row, 'e2e_speedup_vs_dense'):.3f} | {f(row, 'nll_delta_vs_dense'):.4f} | "
            f"{f(row, 'arc_acc_norm'):.4f} | {row.get('report_note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Suggested Points",
            "",
            "- `point_020`: conservative quality point, better NLL and ARC-C than uniform dense NVFP4.",
            "- `point_013` and `point_019`: low-loss supporting points that make the frontier trend easier to see.",
            "- `point_024`: main point, faster than uniform sparse baselines with much better NLL and ARC-C.",
            "- `point_026`: aggressive point, still much better quality than uniform sparse baselines.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def label(row: dict[str, Any]) -> str:
    return row["label"].replace("point_", "P").replace("all_", "")


def parse_counts(raw: str) -> dict[str, int]:
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): int(value) for key, value in parsed.items()}


if __name__ == "__main__":
    main()
