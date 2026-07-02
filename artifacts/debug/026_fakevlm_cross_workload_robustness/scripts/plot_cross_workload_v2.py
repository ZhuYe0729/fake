#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = SCRIPT_DIR.parents[0]

SCENARIO_LABELS = {
    "prefill_only": "Prefill",
    "normal_01": "Prefill+Decoding",
    "normal_02": "Decode-Heavy",
    "geomean": "Avg.",
}

METHOD_LABELS = {
    "dense_bf16": "Dense\nBF16",
    "uniform_dense_nvfp4": "W4A4",
    "uniform_sparse_bf16": "2:4\nBF16",
    "uniform_sparse_nvfp4": "2:4\nW4A4",
    "uniform_marlin_weight_only": "W4A16",
    "our_linear_hybrid": "Ours",
}

METHODS = tuple(METHOD_LABELS)
SCENARIOS = ("prefill_only", "normal_01", "normal_02", "geomean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FakeVLM cross-workload speedup v2.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    cache_dir = Path("/tmp/cospaq_matplotlib_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()
    rows = read_rows(args.output_root / "summary" / "workload_method_table.csv")
    values = np.array(
        [
            [float(rows[scenario][f"{method}_speedup_vs_dense"]) for method in METHODS]
            for scenario in SCENARIOS
        ]
    )

    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    x = np.arange(len(SCENARIOS))
    width = 0.12
    colors = ["#8c97a5", "#b7bec8", "#718497", "#9da9b5", "#566575", "#d24b35"]
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    for index, method in enumerate(METHODS):
        bars = ax.bar(
            x + offsets[index],
            values[:, index],
            width=width,
            color=colors[index],
            label=METHOD_LABELS[method].replace("\n", " "),
        )
        if method == "our_linear_hybrid":
            for bar, value in zip(bars, values[:, index]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.035,
                    f"{value:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

    ax.axhline(1.0, color="#2f3440", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_ylabel("Speedup vs dense BF16")
    ax.set_title("FakeVLM Cross-Workload Robustness")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[item] for item in SCENARIOS])
    ax.set_ylim(0, max(values.flatten()) * 1.24)
    ax.grid(axis="y", color="#d7dbe2", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()

    output_base = args.output_root / "summary" / "fakevlm_cross_workload_v2_speedup"
    fig.savefig(output_base.with_suffix(".png"), dpi=240)
    fig.savefig(output_base.with_suffix(".pdf"))
    print(f"wrote {output_base.with_suffix('.png')}")
    print(f"wrote {output_base.with_suffix('.pdf')}")


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


if __name__ == "__main__":
    main()
