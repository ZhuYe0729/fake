#!/usr/bin/env python3
"""Build a compact, reproducible diagnostic report for the v2 NLL proxy."""
from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt

from common import DEBUG


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = DEBUG / "report"
    rows = read_csv(report / "predictions.csv")
    metrics = json.loads((report / "metrics.json").read_text())
    style = {"old_train": ("#4c78a8", "o"), "old_holdout": ("#e45756", "s"), "mechanism_train": ("#59a14f", "o"), "mechanism_holdout": ("#f28e2b", "s")}
    for group in style:
        subset = [row for row in rows if row["group"] == group]
        if subset:
            color, marker = style[group]
            plt.scatter([float(row["actual_delta_nll"]) for row in subset], [float(row["v2_predicted_delta_nll"]) for row in subset], color=color, marker=marker, label=group, alpha=.85)
    maximum = max(max(float(row[key]) for row in rows) for key in ("actual_delta_nll", "v2_predicted_delta_nll"))
    plt.plot([0, maximum], [0, maximum], "--", color="#555555", linewidth=1)
    plt.xlabel("Measured real-vLLM ΔNLL vs dense BF16")
    plt.ylabel("V2 predicted ΔNLL")
    plt.title("Llama2 prefill: mechanism-aware quality calibration")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(report / "v2_predicted_vs_measured.png", dpi=200); plt.close()

    for group in style:
        subset = [row for row in rows if row["group"] == group]
        if subset:
            color, marker = style[group]
            plt.scatter(range(len(subset)), [float(row["v2_residual"]) for row in subset], color=color, marker=marker, label=group)
    plt.axhline(0, color="#555555", linewidth=1)
    plt.xlabel("Policy index within split")
    plt.ylabel("Measured − V2 predicted ΔNLL")
    plt.title("V2 residuals by held-out calibration family")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(report / "v2_residuals.png", dpi=200); plt.close()

    lines = ["# Llama2 prefill mechanism-aware NLL proxy", "", "All labels are direct phase-heterogeneous vLLM prompt-logprob NLL on the fixed 100×2048 WikiText blocks used by `046`.  The fit has zero BF16 bias and non-negative quantization, sparsity, sparse-accumulation, and sparse–quantization-interaction terms.", "", "## Validation", "", "| split | MAE | RMSE | signed error | Spearman |", "|---|---:|---:|---:|---:|"]
    for group, value in metrics["v2"].items():
        lines.append(f"| {group} | {value['mae']:.6f} | {value['rmse']:.6f} | {value['signed_error']:.6f} | {value['spearman']:.4f} |")
    lines += ["", "## Old holdout comparison", "", "| model | MAE | RMSE | signed error | Spearman |", "|---|---:|---:|---:|---:|"]
    for name, value in metrics["old_holdout_comparison"].items():
        lines.append(f"| {name} | {value['mae']:.6f} | {value['rmse']:.6f} | {value['signed_error']:.6f} | {value['spearman']:.4f} |")
    lines += ["", "## Per-policy calibration", "", "| policy | split | measured ΔNLL | V2 predicted ΔNLL | residual | V1 prediction (where available) |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        v1 = "" if not row["v1_predicted_delta_nll"] else f"{float(row['v1_predicted_delta_nll']):.6f}"
        lines.append(f"| {row['policy_id']} | {row['group']} | {float(row['actual_delta_nll']):.6f} | {float(row['v2_predicted_delta_nll']):.6f} | {float(row['v2_residual']):.6f} | {v1} |")
    (report / "summary.md").write_text("\n".join(lines) + "\n")
    print(report / "summary.md")


if __name__ == "__main__":
    main()
