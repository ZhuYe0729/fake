#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt


PROJECT_ROOT = ROOT.parents[2]
SOURCE_CSV = PROJECT_ROOT / "artifacts" / "debug" / "024_fakevlm_prefill_global_pareto" / "costs" / "batch_16" / "module_method_candidates.csv"
METHOD = "sparse_nvfp4"


def main() -> None:
    rows = read_source_rows()
    summary = build_layer_summary(rows)
    write_csv(ROOT / "sparse_nvfp4_layer_profile_only.csv", summary)
    plot(summary)
    print(f"wrote layer-profile-only outputs to {ROOT}")


def read_source_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["method"] == METHOD and row.get("supported", "True") == "True"]


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def build_layer_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(f(row, "layer"))].append(row)
    out = []
    for layer, items in sorted(grouped.items()):
        local_errors = [f(row, "output_rel_mse") for row in items]
        quality_costs = [f(row, "quality_cost") for row in items]
        out.append(
            {
                "layer": layer,
                "modules": len(items),
                "mean_output_rel_mse": mean(local_errors),
                "mean_quality_cost": mean(quality_costs),
                "normalized_mean_output_rel_mse": mean(local_errors),
                "normalized_mean_quality_cost": mean(quality_costs),
            }
        )
    error_median = median(float(row["mean_output_rel_mse"]) for row in out)
    quality_median = median(float(row["mean_quality_cost"]) for row in out)
    for row in out:
        row["normalized_mean_output_rel_mse"] = float(row["mean_output_rel_mse"]) / error_median
        row["normalized_mean_quality_cost"] = float(row["mean_quality_cost"]) / quality_median
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]]) -> None:
    layers = [int(row["layer"]) for row in rows]
    error = [float(row["normalized_mean_output_rel_mse"]) for row in rows]
    quality = [float(row["normalized_mean_quality_cost"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.plot(layers, error, marker="o", markersize=4.2, linewidth=1.9, color="#4c78a8", label="Mean local output relative MSE / median")
    ax.plot(layers, quality, marker="s", markersize=4.2, linewidth=1.9, color="#e45756", label="Mean NLL-delta proxy / median")
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_title("FakeVLM Sparse NVFP4: layer sensitivity expands the loss gap")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Normalized layer mean")
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    ax.grid(True, linewidth=0.55, alpha=0.35)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.savefig(ROOT / "fakevlm_layer_profile_error_vs_nll_proxy.png", dpi=240)
    fig.savefig(ROOT / "fakevlm_layer_profile_error_vs_nll_proxy.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
