#!/usr/bin/env python3
"""Plot a beautified v2 version of the quality prediction-vs-actual figure.

Changes from the original:
1. Remove the middle NLL delta subplot.
2. Drop the two rightmost outlier points (22 and 25).
3. Top subplot y-axis starts from 0 instead of ~0.6.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "prediction_vs_actual/corrected_nll_batch16_refined_sparse_bf16"
OUT_DIR = DATA_DIR  # save alongside the original
CSV_PATH = DATA_DIR / "quality_comparison_prediction_vs_actual.csv"
BATCH = 16
DENSE_NLL = 0.6241345918
OUTLIER_POINTS = {22, 25}  # point indices to drop


def load_rows(csv_path: Path, batch: int) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["batch_size"]) == batch:
                rows.append(row)
    return rows


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "figure.dpi": 150,
    })

    all_rows = load_rows(CSV_PATH, BATCH)
    rows = [r for r in all_rows if int(r["point_index"]) not in OUTLIER_POINTS]
    dropped = [r for r in all_rows if int(r["point_index"]) in OUTLIER_POINTS]
    print(f"Kept {len(rows)} points, dropped {len(dropped)} outlier points: "
          f"{[int(r['point_index']) for r in dropped]}")

    points = [int(r["point_index"]) for r in rows]
    predicted_nll = [float(r["predicted_nll"]) for r in rows]
    actual_nll = [float(r["actual_nll"]) for r in rows]
    accuracy = [float(r["fakeclue_global_accuracy"]) for r in rows]

    # Compute error stats for annotation
    errors = [abs(p - a) for p, a in zip(predicted_nll, actual_nll)]
    mae = sum(errors) / len(errors)
    max_err = max(errors)

    # --- Two-panel figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1.6, 1]})

    # Color palette
    pred_color = "#2563eb"   # blue
    actual_color = "#dc2626"  # red
    dense_color = "#6b7280"  # gray

    # === Top: NLL predicted vs actual ===
    ax1.plot(points, predicted_nll, "o-", color=pred_color, linewidth=2.2,
             markersize=9, label="Predicted NLL", zorder=3)
    ax1.plot(points, actual_nll, "s-", color=actual_color, linewidth=2.2,
             markersize=9, label="Actual NLL", zorder=3)
    ax1.axhline(y=DENSE_NLL, color=dense_color, linewidth=1.2, linestyle="--",
                label=f"Dense baseline ({DENSE_NLL:.3f})", zorder=2)

    # Fill between to show prediction error area
    ax1.fill_between(points, predicted_nll, actual_nll, alpha=0.08,
                     color="#7c3aed", label="Prediction error area")

    ax1.set_ylabel("NLL")
    ax1.set_ylim(0, None)
    ax1.legend(loc="lower left", framealpha=0.9, edgecolor="#d1d5db")
    ax1.grid(True, color="#e5e7eb", linewidth=0.6)
    ax1.set_title(f"FakeVLM Quality Prediction vs Actual  (batch {BATCH})")

    # Annotate MAE
    ax1.text(0.98, 0.04, f"MAE = {mae:.4f}  |  max error = {max_err:.4f}",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=10, color="#374151",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9fafb", edgecolor="#d1d5db", alpha=0.85))

    # === Bottom: FakeClue accuracy ===
    ax2.plot(points, accuracy, "o-", color="#047857", linewidth=2.2,
             markersize=9, zorder=3)
    # Baseline accuracy at point 0
    dense_acc = accuracy[0] if points[0] == 0 else None
    if dense_acc is not None:
        ax2.axhline(y=dense_acc, color=dense_color, linewidth=1.2, linestyle="--",
                    label=f"Dense accuracy ({dense_acc:.4f})", zorder=2)
    ax2.set_ylabel("FakeClue Accuracy")
    ax2.set_xlabel("Strategy Point")
    ax2.legend(loc="lower left", framealpha=0.9, edgecolor="#d1d5db")
    ax2.grid(True, color="#e5e7eb", linewidth=0.6)

    # Format y-axis to show more precision
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Hide x-axis tick labels
    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="x", labelbottom=False)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"quality_batch_{BATCH}_prediction_vs_actual_v2.{ext}"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        print(f"Saved {out_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()