#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = SCRIPT_DIR.parents[0]

METHOD_LABELS = {
    "dense_bf16": "Dense BF16",
    "uniform_dense_nvfp4": "Uniform W4A4",
    "uniform_sparse_bf16": "Uniform 2:4 BF16",
    "uniform_sparse_nvfp4": "Uniform 2:4 W4A4",
    "uniform_marlin_weight_only": "Uniform W4A16",
    "uniform_dense_nvfp4_prefill_marlin_decode": "Uniform W4A4/W4A16",
    "our_linear_hybrid": "Ours",
}

METHODS = tuple(METHOD_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FakeVLM average cross-workload speedup.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--average", choices=("geomean", "arith_mean"), default="geomean")
    return parser.parse_args()


def main() -> None:
    cache_dir = Path("/tmp/cospaq_matplotlib_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib.pyplot as plt

    args = parse_args()
    table_path = args.output_root / "summary" / "workload_method_table.csv"
    row = average_row(table_path, args.average)
    values = [float(row[f"{method}_speedup_vs_dense"]) for method in METHODS]
    labels = [METHOD_LABELS[method] for method in METHODS]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    colors = ["#7a8797"] * len(METHODS)
    colors[-1] = "#d24b35"
    bars = ax.bar(range(len(METHODS)), values, color=colors, width=0.68)
    ax.axhline(1.0, color="#2f3440", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylabel("Average speedup vs dense BF16")
    ax.set_title("FakeVLM Cross-Workload Average Speedup")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", color="#d7dbe2", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.025,
            f"{value:.3f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()

    output_base = args.output_root / "summary" / f"fakevlm_cross_workload_average_{args.average}_speedup"
    fig.savefig(output_base.with_suffix(".png"), dpi=220)
    fig.savefig(output_base.with_suffix(".pdf"))
    print(f"wrote {output_base.with_suffix('.png')}")
    print(f"wrote {output_base.with_suffix('.pdf')}")


def average_row(path: Path, average: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scenario"] == average:
                return row
    raise RuntimeError(f"missing average row {average}: {path}")


if __name__ == "__main__":
    main()
