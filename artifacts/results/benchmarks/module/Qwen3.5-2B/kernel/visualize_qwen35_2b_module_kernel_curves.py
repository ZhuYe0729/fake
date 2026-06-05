#!/usr/bin/env python
"""Plot Qwen3.5-2B module-forward kernel latency curves.

Each subplot is one Qwen3.5-2B compressible Linear shape. Curves show packaged
Linear.forward latency as token M grows.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "qwen35_2b_module_kernel_curves.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "qwen35_2b_module_kernel_latency_curves.png"

KERNELS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]
KERNEL_LABELS = {
    "dense_bf16": "BF16",
    "dense_nvfp4": "Dense NVFP4",
    "sparse_bf16": "Sparse BF16",
    "sparse_nvfp4": "Sparse NVFP4",
    "marlin_nvfp4": "Marlin NVFP4",
}
KERNEL_COLORS = {
    "dense_bf16": "#1f77b4",
    "dense_nvfp4": "#2ca02c",
    "sparse_bf16": "#ff7f0e",
    "sparse_nvfp4": "#d62728",
    "marlin_nvfp4": "#9467bd",
}
KERNEL_MARKERS = {
    "dense_bf16": "o",
    "dense_nvfp4": "s",
    "sparse_bf16": "^",
    "sparse_nvfp4": "D",
    "marlin_nvfp4": "P",
}

LINEAR_ORDER = [
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _collect(rows: list[dict[str, str]]) -> tuple[dict, dict]:
    data: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    meta: dict[str, dict[str, int]] = {}
    for row in rows:
        group = row["linear_group"]
        meta[group] = {
            "count": int(row["linear_count"]),
            "n": int(row["n"]),
            "k": int(row["k"]),
        }
        if row["status"] != "pass" or not row["latency_ms"]:
            continue
        data[(group, row["kernel"])].append((int(row["m"]), float(row["latency_ms"])))

    for points in data.values():
        points.sort(key=lambda item: item[0])
    return data, meta


def _plot_latency_curves(rows: list[dict[str, str]], output: Path, *, title: str) -> None:
    data, meta = _collect(rows)
    groups = [group for group in LINEAR_ORDER if group in meta]
    if not groups:
        raise RuntimeError("No Qwen3.5-2B linear groups found in CSV")

    fig, axes = plt.subplots(3, 4, figsize=(28, 17), sharex=True)
    axes_flat = axes.ravel()

    global_latencies = [
        latency
        for points in data.values()
        for _, latency in points
        if np.isfinite(latency) and latency > 0
    ]
    y_min = min(global_latencies) * 0.75
    y_max = max(global_latencies) * 1.35

    for ax, group in zip(axes_flat, groups):
        group_meta = meta[group]
        for kernel in KERNELS:
            points = data.get((group, kernel), [])
            if not points:
                continue
            xs = [m for m, _ in points]
            ys = [latency for _, latency in points]
            ax.plot(
                xs,
                ys,
                label=KERNEL_LABELS[kernel],
                color=KERNEL_COLORS[kernel],
                marker=KERNEL_MARKERS[kernel],
                linewidth=1.8,
                markersize=4.5,
                alpha=0.92,
            )

        ax.set_title(
            f"{group}\ncount={group_meta['count']}, N={group_meta['n']}, K={group_meta['k']}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="major", linestyle="-", alpha=0.25)
        ax.grid(True, which="minor", linestyle=":", alpha=0.15)
        ax.tick_params(axis="both", labelsize=9)

    for ax in axes_flat[len(groups) :]:
        ax.axis("off")

    m_values = sorted({int(row["m"]) for row in rows})
    for ax in axes[-1, :]:
        ax.set_xlabel("M (tokens)", fontsize=11)
        ax.set_xticks(m_values)
        ax.set_xticklabels([str(m) for m in m_values], rotation=45, ha="right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latency (ms, log scale)", fontsize=11)

    handles = [
        Line2D(
            [0],
            [0],
            label=KERNEL_LABELS[kernel],
            color=KERNEL_COLORS[kernel],
            marker=KERNEL_MARKERS[kernel],
            linewidth=1.8,
            markersize=5,
        )
        for kernel in KERNELS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {output}")
    print(f"saved {pdf_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--title",
        default="Qwen3.5-2B Packaged Linear.forward Latency Curves by Shape",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.input)
    _plot_latency_curves(rows, args.output, title=args.title)


if __name__ == "__main__":
    main()
