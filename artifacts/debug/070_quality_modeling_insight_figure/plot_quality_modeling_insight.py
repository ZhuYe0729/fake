#!/usr/bin/env python3
"""Build the paper quality-modeling insight figure from frozen Llama-2 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling/summary/"
    "kernel_prefill_loss_modeling/layer_type_kernel_local_error_loss_joined.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_PAPER_DIR = REPO_ROOT / "references/Figures/draft2"
DEFAULT_EXISTING_INCLUDE_DIR = REPO_ROOT / "references/Figures/draft"

METHOD = "sparse_nvfp4"
TYPE_ORDER = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
TYPE_LABELS = {
    "q_proj": "Q",
    "k_proj": "K",
    "v_proj": "V",
    "o_proj": "O",
    "gate_proj": "Gate",
    "up_proj": "Up",
    "down_proj": "Down",
}
TYPE_COLORS = {
    "q_proj": "#4C78A8",
    "k_proj": "#72B7B2",
    "v_proj": "#F58518",
    "o_proj": "#54A24B",
    "gate_proj": "#B279A2",
    "up_proj": "#E45756",
    "down_proj": "#9D755D",
}

DOUBLE_COLUMN_WIDTH = 7.0
FIGURE_HEIGHT = 2.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-output-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument(
        "--existing-include-dir", type=Path, default=DEFAULT_EXISTING_INCLUDE_DIR
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["method"] != METHOD or raw["policy_kind"] != "layer_type":
                continue
            row = {
                "layer": int(raw["layer"]),
                "linear_type": raw["linear_type"],
                "module_name": raw["module_name"],
                "local_error": float(raw["output_rel_mse"]),
                "delta_nll": float(raw["loss_delta_vs_dense"]),
            }
            if row["linear_type"] not in TYPE_ORDER:
                raise ValueError(f"Unexpected projection type: {row['linear_type']}")
            if not math.isfinite(row["local_error"]) or not math.isfinite(row["delta_nll"]):
                raise ValueError(f"Non-finite row: {row}")
            rows.append(row)

    expected = 32 * len(TYPE_ORDER)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {METHOD} single-Linear rows, got {len(rows)}")
    if len({str(row["module_name"]) for row in rows}) != expected:
        raise ValueError("Single-Linear rows do not have unique module names")
    return rows


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator


def summarize_layer_profile(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)

    profile: list[dict[str, object]] = []
    for layer in range(32):
        selected = grouped[layer]
        if len(selected) != len(TYPE_ORDER):
            raise ValueError(
                f"Expected {len(TYPE_ORDER)} projections for layer={layer}; got {len(selected)}"
            )
        profile.append(
            {
                "layer": layer,
                "projections": len(selected),
                "mean_local_error": mean(float(row["local_error"]) for row in selected),
                "mean_delta_nll": mean(float(row["delta_nll"]) for row in selected),
            }
        )

    local_median = median(float(row["mean_local_error"]) for row in profile)
    delta_median = median(float(row["mean_delta_nll"]) for row in profile)
    if local_median <= 0 or delta_median <= 0:
        raise ValueError("Layer-profile medians must be positive for normalization")
    for row in profile:
        row["normalized_mean_local_error"] = (
            float(row["mean_local_error"]) / local_median
        )
        row["normalized_mean_delta_nll"] = (
            float(row["mean_delta_nll"]) / delta_median
        )
    return profile


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 10.0,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.9,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_figure(
    rows: list[dict[str, object]],
    profile: list[dict[str, object]],
    output_paths: list[Path],
) -> dict[str, float]:
    xs = [float(row["local_error"]) for row in rows]
    ys = [float(row["delta_nll"]) for row in rows]
    single_linear_pearson = pearson(xs, ys)
    single_linear_spearman = pearson(average_ranks(xs), average_ranks(ys))
    profile_local = [float(row["mean_local_error"]) for row in profile]
    profile_delta = [float(row["mean_delta_nll"]) for row in profile]
    profile_pearson = pearson(profile_local, profile_delta)
    profile_spearman = pearson(
        average_ranks(profile_local), average_ranks(profile_delta)
    )

    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        sharex=True,
        figsize=(DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.34},
    )

    profile_ax = axes[0]
    layers = [int(row["layer"]) for row in profile]
    profile_ax.plot(
        layers,
        [float(row["normalized_mean_local_error"]) for row in profile],
        color="#4C78A8",
        marker="o",
        label="Local output error",
    )
    profile_ax.plot(
        layers,
        [float(row["normalized_mean_delta_nll"]) for row in profile],
        color="#E45756",
        marker="s",
        label=r"Measured $\Delta$NLL",
    )
    profile_ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
    profile_ax.set_xlabel("Transformer layer", labelpad=-1)
    profile_ax.set_ylabel("Normalized layer mean")
    profile_ax.set_xlim(-0.8, 31.8)
    profile_ax.set_xticks([0, 5, 10, 15, 20, 25, 31])
    profile_ax.grid(True, color="#D0D0D0", linewidth=0.55, alpha=0.65)
    profile_ax.set_axisbelow(True)
    profile_ax.legend(loc="upper left", frameon=False, handlelength=1.5)

    type_ax = axes[1]
    for linear_type in TYPE_ORDER:
        selected = [row for row in rows if str(row["linear_type"]) == linear_type]
        selected.sort(key=lambda row: int(row["layer"]))
        type_ax.plot(
            [int(row["layer"]) for row in selected],
            [float(row["delta_nll"]) for row in selected],
            label=TYPE_LABELS[linear_type],
            color=TYPE_COLORS[linear_type],
            marker="o",
        )
    type_ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
    type_ax.set_xlabel("Transformer layer", labelpad=-1)
    type_ax.set_ylabel(r"Measured $\Delta$NLL")
    type_ax.set_xlim(-0.8, 31.8)
    type_ax.set_xticks([0, 5, 10, 15, 20, 25, 31])
    type_ax.grid(True, color="#D0D0D0", linewidth=0.55, alpha=0.65)
    type_ax.set_axisbelow(True)
    type_ax.legend(
        loc="upper center",
        ncol=4,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.25,
        columnspacing=0.65,
    )

    profile_ax.text(
        0.5,
        -0.17,
        "(a) Local error fails to predict impact",
        ha="center",
        va="top",
        transform=profile_ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
    )
    type_ax.text(
        0.5,
        -0.17,
        "(b) Sensitivity depends on depth and type",
        ha="center",
        va="top",
        transform=type_ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
    )
    for ax in axes:
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("semibold")
    fig.subplots_adjust(top=0.98, bottom=0.20, left=0.105, right=0.995)

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "single_linear_pearson": single_linear_pearson,
        "single_linear_spearman": single_linear_spearman,
        "layer_profile_pearson": profile_pearson,
        "layer_profile_spearman": profile_spearman,
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    profile: list[dict[str, object]],
    metrics: dict[str, float],
    source: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sparse_nvfp4_single_linear_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "layer_profile_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile[0]))
        writer.writeheader()
        writer.writerows(profile)
    metadata = {
        "source_csv": str(source.resolve()),
        "method": METHOD,
        "model": "Llama-2-7B",
        "dataset": "WikiText-2 train calibration samples",
        "samples": 128,
        "sequence_length": 512,
        "single_linear_rows": len(rows),
        "local_error_metric": "output_rel_mse",
        "global_impact_metric": "loss_delta_vs_dense",
        **metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    profile = summarize_layer_profile(rows)
    artifact_stem = args.output_dir / "quality_modeling_insight"
    paper_stem = args.paper_output_dir / "quality_modeling_insight"
    existing_include_stem = args.existing_include_dir / "quality"
    metrics = draw_figure(
        rows,
        profile,
        [
            artifact_stem.with_suffix(".pdf"),
            artifact_stem.with_suffix(".png"),
            paper_stem.with_suffix(".pdf"),
            paper_stem.with_suffix(".png"),
            existing_include_stem.with_suffix(".pdf"),
            existing_include_stem.with_suffix(".png"),
        ],
    )
    write_outputs(args.output_dir, rows, profile, metrics, args.input_csv)


if __name__ == "__main__":
    main()
