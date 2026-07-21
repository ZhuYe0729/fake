#!/usr/bin/env python3
"""Create compact calibration tables and diagnostic plots for one model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from common import MODELS, model_root


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    root = model_root(args.model)
    report = root / "reports/quality"
    rows = read_csv(report / "predictions.csv")
    metrics = json.loads((report / "metrics.json").read_text())
    for split, color, marker in (("train", "#4c78a8", "o"), ("holdout", "#e45756", "s")):
        subset = [row for row in rows if row["split"] == split]
        plt.scatter([float(row["actual_delta_nll"]) for row in subset], [float(row["predicted_delta_nll"]) for row in subset], color=color, marker=marker, label=split, alpha=.8)
    maximum = max(max(float(row[key]) for row in rows) for key in ("actual_delta_nll", "predicted_delta_nll"))
    plt.plot([0, maximum], [0, maximum], "--", color="#555555", linewidth=1)
    plt.xlabel("Measured real-vLLM ΔNLL vs dense BF16")
    plt.ylabel("Predicted ΔNLL")
    plt.title(f"{args.model}: real-vLLM quality proxy calibration")
    plt.legend(); plt.tight_layout(); plt.savefig(report / "predicted_vs_measured.png", dpi=180); plt.close()
    plt.scatter([float(row["count_sparse_nvfp4"]) + float(row["count_sparse_bf16"]) for row in rows], [float(row["residual_actual_minus_predicted"]) for row in rows], c=["#e45756" if row["split"] == "holdout" else "#4c78a8" for row in rows], alpha=.8)
    plt.axhline(0, color="#555555", linewidth=1)
    plt.xlabel("Sparse fused-linear modules")
    plt.ylabel("Measured − predicted ΔNLL")
    plt.title(f"{args.model}: residual versus sparse usage")
    plt.tight_layout(); plt.savefig(report / "residual_vs_sparse_count.png", dpi=180); plt.close()
    lines = [f"# {args.model} real-vLLM prefill quality calibration", "", "The labels are direct vLLM prompt-logprob NLL over 100 fixed WikiText blocks; local-error features are retained from the prior model.", "", "## Metrics", "", "| split | MAE | RMSE | signed error | Spearman |", "|---|---:|---:|---:|---:|"]
    for split in ("train", "holdout"):
        row = metrics["metrics"][split]
        lines.append(f"| {split} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['mean_signed_error']:.6f} | {row['spearman']:.4f} |")
    lines += ["", "## Per-policy predictions", "", "| policy | split | kind | measured ΔNLL | predicted ΔNLL | residual |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['policy_id']} | {row['split']} | {row['policy_kind']} | {float(row['actual_delta_nll']):.6f} | {float(row['predicted_delta_nll']):.6f} | {float(row['residual_actual_minus_predicted']):.6f} |")
    (report / "summary.md").write_text("\n".join(lines) + "\n")
    print(report / "summary.md")


if __name__ == "__main__":
    main()
