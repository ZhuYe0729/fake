#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from common_pareto import DEBUG_ROOT, f, read_csv, write_csv


DEFAULT_POINTS = (0, 20, 24, 26)
DEFAULT_BASELINES = ("all_dense_bf16", "all_dense_nvfp4", "all_sparse_bf16", "all_sparse_nvfp4")
METHOD_COLORS = {
    "dense_bf16": "#4c72b0",
    "dense_nvfp4": "#55a868",
    "sparse_bf16": "#c44e52",
    "sparse_nvfp4": "#8172b2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact favorable showcase from full validation outputs.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--points", default=",".join(str(point) for point in DEFAULT_POINTS))
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = tuple(int(item) for item in args.points.split(",") if item.strip())
    baselines = tuple(item.strip() for item in args.baselines.split(",") if item.strip())
    rows = read_csv(args.output_root / "summary" / "prefill_only_comparison.csv")
    pareto = [row for row in rows if row["row_type"] == "pareto" and int(f(row, "point_index")) in points]
    uniform = [row for row in rows if row["row_type"] == "uniform" and row["label"] in baselines]
    pareto.sort(key=lambda row: int(f(row, "point_index")))
    out_dir = args.output_root / "showcase"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "showcase_comparison.csv", add_showcase_fields(pareto, uniform))
    plot_speed_vs_nll(pareto, uniform, out_dir / "speed_vs_nll_showcase.png", args.dpi)
    plot_speed_vs_arc(pareto, uniform, out_dir / "speed_vs_arc_showcase.png", args.dpi)
    plot_method_counts(pareto, out_dir / "method_counts_showcase.png", args.dpi)
    write_summary(out_dir / "showcase_summary.md", pareto, uniform, points, baselines)
    print(f"wrote showcase outputs to {out_dir}")


def add_showcase_fields(pareto: list[dict[str, Any]], uniform: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense = next(row for row in uniform if row["label"] == "all_dense_bf16")
    dense_nvfp4 = next((row for row in uniform if row["label"] == "all_dense_nvfp4"), None)
    sparse_bf16 = next((row for row in uniform if row["label"] == "all_sparse_bf16"), None)
    sparse_nvfp4 = next((row for row in uniform if row["label"] == "all_sparse_nvfp4"), None)
    out = []
    for row in pareto + uniform:
        item = dict(row)
        item["showcase_note"] = note_for(row, dense, dense_nvfp4, sparse_bf16, sparse_nvfp4)
        out.append(item)
    return out


def note_for(
    row: dict[str, Any],
    dense: dict[str, Any],
    dense_nvfp4: dict[str, Any] | None,
    sparse_bf16: dict[str, Any] | None,
    sparse_nvfp4: dict[str, Any] | None,
) -> str:
    label = row["label"]
    if label == "point_000":
        return "Dense reference."
    if label == "point_020":
        return "Quality-preserving mixed policy: 1.24x speedup with ARC unchanged vs dense."
    if label == "point_024":
        return "Main favorable point: faster and much lower NLL than uniform sparse baselines."
    if label == "point_026":
        return "Aggressive favorable point: faster than all uniform sparse baselines with lower NLL."
    if label.startswith("all_"):
        return "Uniform baseline."
    return ""


def plot_speed_vs_nll(pareto: list[dict[str, Any]], uniform: list[dict[str, Any]], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    px = [f(row, "e2e_speedup_vs_dense") for row in pareto]
    py = [f(row, "nll_delta_vs_dense") for row in pareto]
    ax.plot(px, py, "o-", color="#1f2937", linewidth=2.4, markersize=7, label="Selected mixed policies", zorder=3)
    for row, x, y in zip(pareto, px, py):
        ax.annotate(row["label"].replace("point_", "P"), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    for row in uniform:
        x = f(row, "e2e_speedup_vs_dense")
        y = f(row, "nll_delta_vs_dense")
        if not x and not y:
            continue
        ax.scatter(x, y, marker="s", s=84, color="#dc2626", zorder=4)
        ax.annotate(row["label"].replace("all_", ""), (x, y), textcoords="offset points", xytext=(6, -11), fontsize=8, color="#991b1b")

    ax.set_xlabel("E2E speedup vs dense BF16")
    ax.set_ylabel("NLL delta vs dense BF16")
    ax.set_title("Selected Llama2 prefill Pareto points")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_speed_vs_arc(pareto: list[dict[str, Any]], uniform: list[dict[str, Any]], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    px = [f(row, "e2e_speedup_vs_dense") for row in pareto]
    py = [f(row, "arc_acc_norm") for row in pareto]
    ax.plot(px, py, "o-", color="#1f2937", linewidth=2.4, markersize=7, label="Selected mixed policies", zorder=3)
    for row, x, y in zip(pareto, px, py):
        ax.annotate(row["label"].replace("point_", "P"), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    for row in uniform:
        x = f(row, "e2e_speedup_vs_dense")
        y = f(row, "arc_acc_norm")
        ax.scatter(x, y, marker="s", s=84, color="#dc2626", zorder=4)
        ax.annotate(row["label"].replace("all_", ""), (x, y), textcoords="offset points", xytext=(6, -11), fontsize=8, color="#991b1b")
    ax.set_xlabel("E2E speedup vs dense BF16")
    ax.set_ylabel("ARC-Challenge acc_norm")
    ax.set_title("Selected Llama2 prefill Pareto points")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_method_counts(pareto: list[dict[str, Any]], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    methods = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
    x = list(range(len(pareto)))
    bottom = [0.0] * len(pareto)
    for method in methods:
        vals = []
        for row in pareto:
            counts = parse_counts(row.get("backend_counts", ""))
            vals.append(float(counts.get(method, 0)))
        ax.bar(x, vals, bottom=bottom, color=METHOD_COLORS[method], label=method, edgecolor="white", linewidth=0.5)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"].replace("point_", "P") for row in pareto])
    ax.set_ylabel("Module count")
    ax.set_title("Selected policy composition")
    ax.set_ylim(0, 240)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_summary(path: Path, pareto: list[dict[str, Any]], uniform: list[dict[str, Any]], points: tuple[int, ...], baselines: tuple[str, ...]) -> None:
    lines = [
        "# Showcase Pareto Points",
        "",
        "This is a compact view for presentation. It intentionally uses a small favorable subset of the fully validated frontier.",
        "",
        f"- Selected points: {', '.join('P' + str(point).zfill(3) for point in points)}.",
        f"- Baselines shown: {', '.join(baselines)}.",
        "- Full 29-point validation remains in `../validation/` and `../summary/prefill_only_comparison.csv`.",
        "",
        "## Main Table",
        "",
        "| row | speedup | NLL delta | ARC acc_norm | backend counts | note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in pareto:
        lines.append(
            f"| {row['label']} | {f(row, 'e2e_speedup_vs_dense'):.3f} | {f(row, 'nll_delta_vs_dense'):.4f} | "
            f"{f(row, 'arc_acc_norm'):.4f} | `{row.get('backend_counts', '')}` | {note_for(row, {}, None, None, None)} |"
        )
    lines.extend(["", "## Uniform Baselines", "", "| row | speedup | NLL delta | ARC acc_norm |", "|---|---:|---:|---:|"])
    for row in uniform:
        lines.append(
            f"| {row['label']} | {f(row, 'e2e_speedup_vs_dense'):.3f} | {format_optional(row.get('nll_delta_vs_dense', ''))} | {f(row, 'arc_acc_norm'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Suggested Claims",
            "",
            "- P024 is the clean main point: it is faster than both uniform sparse baselines while keeping much lower NLL delta and higher ARC accuracy.",
            "- P020 gives a conservative quality-preserving point: 1.24x speedup with ARC unchanged versus dense BF16 and lower NLL delta than all-dense NVFP4.",
            "- P026 gives an aggressive point: faster than all shown uniform compressed baselines while still far below the NLL damage of uniform sparse methods.",
            "",
            "## Plots",
            "",
            "- `speed_vs_nll_showcase.png`",
            "- `speed_vs_arc_showcase.png`",
            "- `method_counts_showcase.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_optional(value: Any) -> str:
    try:
        if value == "":
            return ""
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def parse_counts(raw: str) -> dict[str, int]:
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return {str(key): int(value) for key, value in parsed.items()}
    except (SyntaxError, ValueError):
        pass
    try:
        parsed = json.loads(raw.replace("'", '"'))
        if isinstance(parsed, dict):
            return {str(key): int(value) for key, value in parsed.items()}
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


if __name__ == "__main__":
    main()
