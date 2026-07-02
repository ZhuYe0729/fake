#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[3]

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

UNIFORM_ACCURACY_METHODS = {
    "dense_bf16": "dense_bf16",
    "uniform_dense_nvfp4": "dense_nvfp4",
    "uniform_sparse_bf16": "sparse_bf16",
    "uniform_sparse_nvfp4": "sparse_nvfp4",
    "uniform_marlin_weight_only": "marlin_weight_only",
}

METHODS = tuple(METHOD_LABELS)
WORKLOAD_SCENARIOS = ("prefill_only", "normal_01", "normal_02")
PLOT_SCENARIOS = WORKLOAD_SCENARIOS + ("geomean",)
PLOT_SCENARIOS_V4 = WORKLOAD_SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FakeVLM cross-workload speedup and accuracy.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument(
        "--uniform-accuracy-csv",
        type=Path,
        default=REPO_ROOT / "artifacts/debug/020_fakevlm_uniform_accuracy/summary/accuracy_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    cache_dir = Path("/tmp/cospaq_matplotlib_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()
    speed_rows = read_rows(args.output_root / "summary" / "workload_method_table.csv", "scenario")
    uniform_accuracy = read_uniform_accuracy(args.uniform_accuracy_csv)
    our_accuracy = read_our_accuracy(args.output_root)

    speed_values = np.array(
        [
            [float(speed_rows[scenario][f"{method}_speedup_vs_dense"]) for method in METHODS]
            for scenario in PLOT_SCENARIOS
        ]
    )
    accuracy_values = np.array(
        [[accuracy_for(method, scenario, uniform_accuracy, our_accuracy) for method in METHODS] for scenario in PLOT_SCENARIOS]
    )

    x = np.arange(len(PLOT_SCENARIOS))
    width = 0.12
    colors = ["#8c97a5", "#b7bec8", "#718497", "#9da9b5", "#566575", "#d24b35"]
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    fig, speed_ax = plt.subplots(figsize=(12.8, 5.8))
    for index, method in enumerate(METHODS):
        bars = speed_ax.bar(
            x + offsets[index],
            speed_values[:, index],
            width=width,
            color=colors[index],
            label=METHOD_LABELS[method].replace("\n", " "),
        )
        if method == "our_linear_hybrid":
            for bar, value in zip(bars, speed_values[:, index]):
                speed_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.035,
                    f"{value:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                )

    speed_ax.axhline(1.0, color="#2f3440", linewidth=1.0, linestyle="--", alpha=0.65)
    speed_ax.set_ylabel("Speedup vs dense BF16")
    speed_ax.set_title("FakeVLM Cross-Workload Robustness")
    speed_ax.set_ylim(0, max(speed_values.flatten()) * 1.24)
    speed_ax.grid(axis="y", color="#d7dbe2", linewidth=0.8, alpha=0.8)
    speed_ax.set_axisbelow(True)
    speed_ax.set_xticks(x)
    speed_ax.set_xticklabels([SCENARIO_LABELS[item] for item in PLOT_SCENARIOS])
    speed_ax.spines["top"].set_visible(False)
    speed_ax.spines["right"].set_visible(False)

    handles, labels = speed_ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.subplots_adjust(bottom=0.18)

    speed_output = args.output_root / "summary" / "fakevlm_cross_workload_v3_speed"
    legacy_output = args.output_root / "summary" / "fakevlm_cross_workload_v3_speed_accuracy"
    for output_base in (speed_output, legacy_output):
        fig.savefig(output_base.with_suffix(".png"), dpi=240)
        fig.savefig(output_base.with_suffix(".pdf"))
        print(f"wrote {output_base.with_suffix('.png')}")
        print(f"wrote {output_base.with_suffix('.pdf')}")
    plt.close(fig)

    v4_speed_values = np.array(
        [
            [float(speed_rows[scenario][f"{method}_speedup_vs_dense"]) for method in METHODS]
            for scenario in PLOT_SCENARIOS_V4
        ]
    )
    v4_x = np.arange(len(PLOT_SCENARIOS_V4))
    v4_fig, v4_ax = plt.subplots(figsize=(10.8, 5.8))
    for index, method in enumerate(METHODS):
        bars = v4_ax.bar(
            v4_x + offsets[index],
            v4_speed_values[:, index],
            width=width,
            color=colors[index],
            label=METHOD_LABELS[method].replace("\n", " "),
        )
        if method == "our_linear_hybrid":
            for bar, value in zip(bars, v4_speed_values[:, index]):
                v4_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.035,
                    f"{value:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                )

    v4_ax.axhline(1.0, color="#2f3440", linewidth=1.0, linestyle="--", alpha=0.65)
    v4_ax.set_ylabel("Speedup vs dense BF16")
    v4_ax.set_title("FakeVLM Cross-Workload Robustness")
    v4_ax.set_ylim(0, max(v4_speed_values.flatten()) * 1.24)
    v4_ax.grid(axis="y", color="#d7dbe2", linewidth=0.8, alpha=0.8)
    v4_ax.set_axisbelow(True)
    v4_ax.set_xticks(v4_x)
    v4_ax.set_xticklabels([SCENARIO_LABELS[item] for item in PLOT_SCENARIOS_V4])
    v4_ax.spines["top"].set_visible(False)
    v4_ax.spines["right"].set_visible(False)

    v4_handles, v4_labels = v4_ax.get_legend_handles_labels()
    v4_fig.legend(v4_handles, v4_labels, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
    v4_fig.subplots_adjust(bottom=0.18)

    v4_speed_output = args.output_root / "summary" / "fakevlm_cross_workload_v4_speed"
    v4_fig.savefig(v4_speed_output.with_suffix(".png"), dpi=240)
    v4_fig.savefig(v4_speed_output.with_suffix(".pdf"))
    print(f"wrote {v4_speed_output.with_suffix('.png')}")
    print(f"wrote {v4_speed_output.with_suffix('.pdf')}")
    plt.close(v4_fig)

    accuracy_methods = METHODS
    accuracy_indices = [METHODS.index(method) for method in accuracy_methods]
    accuracy_width = 0.14
    accuracy_offsets = (np.arange(len(accuracy_methods)) - (len(accuracy_methods) - 1) / 2) * accuracy_width
    acc_fig, acc_ax = plt.subplots(figsize=(12.8, 5.8))
    for offset, method, index in zip(accuracy_offsets, accuracy_methods, accuracy_indices):
        acc_ax.bar(
            x + offset,
            accuracy_values[:, index] * 100.0,
            width=accuracy_width,
            color=colors[index],
            label=METHOD_LABELS[method].replace("\n", " "),
        )

    acc_ax.set_ylabel("Accuracy (%)")
    acc_ax.set_title("FakeVLM Accuracy")
    acc_ax.set_ylim(0.0, 100.0)
    acc_ax.set_yticks([0, 20, 40, 60, 80, 100])
    acc_ax.set_xticks(x)
    acc_ax.set_xticklabels([SCENARIO_LABELS[item] for item in PLOT_SCENARIOS])
    acc_ax.grid(axis="y", color="#d7dbe2", linewidth=0.8, alpha=0.8)
    acc_ax.set_axisbelow(True)
    acc_ax.spines["top"].set_visible(False)
    acc_ax.spines["right"].set_visible(False)
    acc_fig.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
    acc_fig.subplots_adjust(bottom=0.18)

    accuracy_output = args.output_root / "summary" / "fakevlm_cross_workload_v3_accuracy"
    acc_fig.savefig(accuracy_output.with_suffix(".png"), dpi=240)
    acc_fig.savefig(accuracy_output.with_suffix(".pdf"))
    print(f"wrote {accuracy_output.with_suffix('.png')}")
    print(f"wrote {accuracy_output.with_suffix('.pdf')}")


def accuracy_for(
    method: str,
    scenario: str,
    uniform_accuracy: dict[str, float],
    our_accuracy: dict[str, float],
) -> float:
    if method == "our_linear_hybrid":
        if scenario == "geomean":
            return sum(our_accuracy[item] for item in WORKLOAD_SCENARIOS) / len(WORKLOAD_SCENARIOS)
        return our_accuracy[scenario]
    return uniform_accuracy[UNIFORM_ACCURACY_METHODS[method]]


def read_uniform_accuracy(path: Path) -> dict[str, float]:
    rows = read_rows(path, "method")
    return {method: float(row["global_accuracy"]) for method, row in rows.items()}


def read_our_accuracy(output_root: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    missing: list[Path] = []
    for scenario in WORKLOAD_SCENARIOS:
        path = output_root / "accuracy" / scenario / "our_linear_hybrid" / "accuracy.json"
        if not path.exists():
            missing.append(path)
            continue
        import json

        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        result[scenario] = float(data["global_stats"]["global_accuracy"])
    if missing:
        missing_text = "\n".join(str(item) for item in missing)
        raise FileNotFoundError(f"Missing our accuracy files:\n{missing_text}")
    return result


def read_rows(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row[key]: row for row in csv.DictReader(f)}


if __name__ == "__main__":
    main()
