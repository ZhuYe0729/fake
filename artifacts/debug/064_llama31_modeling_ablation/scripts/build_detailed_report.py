#!/usr/bin/env python3
"""Build a richer four-panel paper figure without replacing the compact one."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parents[1]
COLORS = {
    "baseline": "#9ca3af",
    "middle": "#4c78a8",
    "full": "#e45756",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def diagonal(axis, values: list[float]) -> None:
    lower, upper = min(values), max(values)
    margin = 0.06 * max(upper - lower, 1e-12)
    lower = max(0.0, lower - margin)
    upper += margin
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#4b5563", linewidth=1.2, zorder=0)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)


def main() -> None:
    report = OUT / "report"
    report.mkdir(parents=True, exist_ok=True)
    local = read_rows(OUT / "speed/local_predictions.csv")
    speed = read_rows(OUT / "speed/predictions.csv")
    speed_metrics = {row["variant"]: row for row in read_rows(OUT / "speed/metrics.csv")
                     if row["split"] == "holdout"}
    local_metrics = {row["variant"]: row for row in read_rows(OUT / "speed/local_metrics.csv")}
    quality = [row for row in read_rows(OUT / "quality/predictions.csv") if row["split"] == "holdout"]
    quality_metrics = {row["variant"]: row for row in read_rows(OUT / "quality/metrics.csv")
                       if row["split"] == "holdout"}

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))

    # (a) Independent, non-exact Llama3 module shapes.
    axis = axes[0, 0]
    actual = [float(row["measured_ms"]) for row in local]
    roofline = [float(row["predicted_roofline_ms"]) for row in local]
    corrected = [float(row["predicted_shape_corrected_ms"]) for row in local]
    axis.scatter(actual, roofline, color=COLORS["baseline"], marker="o", s=38, alpha=0.85,
                 label=f"Roofline ({100*float(local_metrics['roofline']['mape']):.1f}% MAPE)")
    axis.scatter(actual, corrected, color=COLORS["middle"], marker="s", s=42, alpha=0.9,
                 label=f"+ Shape correction ({100*float(local_metrics['shape_corrected']['mape']):.1f}%)")
    diagonal(axis, actual + roofline + corrected)
    axis.set_xlabel("Measured module latency (ms)")
    axis.set_ylabel("Predicted module latency (ms)")
    axis.set_title("(a) Kernel-level latency")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.2)

    # (b) Per-policy held-out error avoids hiding isotonic ties in a scatter.
    axis = axes[0, 1]
    heldout_speed = [row for row in speed if row["split"] == "holdout"]
    raw_errors, calibrated_errors = [], []
    for row in heldout_speed:
        measured = float(row["measured_e2e_ms"])
        raw_error = 100 * abs(float(row["predicted_shape_corrected_ms"]) - measured) / measured
        calibrated_error = 100 * abs(float(row["predicted_policy_calibrated_ms"]) - measured) / measured
        raw_errors.append(raw_error); calibrated_errors.append(calibrated_error)
        axis.plot([0, 1], [raw_error, calibrated_error], color="#9ca3af", marker="o", linewidth=1.5, alpha=0.85)
        axis.text(1.025, calibrated_error, row["policy_id"], va="center", fontsize=8)
    mean_raw = sum(raw_errors) / len(raw_errors)
    mean_calibrated = sum(calibrated_errors) / len(calibrated_errors)
    axis.plot([0, 1], [mean_raw, mean_calibrated], color=COLORS["full"], marker="D", markersize=7,
              linewidth=3, label=f"Mean: {mean_raw:.1f}% → {mean_calibrated:.1f}%")
    axis.set_xticks((0, 1), ("Raw policy sum", "+ E2E calibration"))
    axis.set_xlim(-0.18, 1.18)
    axis.set_ylabel("Absolute relative E2E error (%)")
    axis.set_title("(b) Held-out policy error")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=9)

    # (c) Only held-out quality policies are shown.
    axis = axes[1, 0]
    measured_quality = [float(row["measured_delta_nll"]) for row in quality]
    quality_variants = (
        ("local_sum", "predicted_local_sum", COLORS["baseline"], "o", "Local sum"),
        ("structure_aware", "predicted_structure_aware", COLORS["middle"], "s", "+ Structure-aware"),
        ("full_phase_aware", "predicted_full_phase_aware", COLORS["full"], "^", "+ Phase-aware"),
    )
    all_quality_values = list(measured_quality)
    for name, field, color, marker, label in quality_variants:
        predicted = [float(row[field]) for row in quality]
        all_quality_values.extend(predicted)
        axis.scatter(measured_quality, predicted, color=color, marker=marker, s=42, alpha=0.82,
                     label=f"{label} (MAE {float(quality_metrics[name]['mae']):.3f})")
    diagonal(axis, all_quality_values)
    axis.set_xlabel("Measured ΔNLL")
    axis.set_ylabel("Predicted ΔNLL")
    axis.set_title("(c) Held-out quality prediction")
    axis.legend(frameon=False, fontsize=8.5)
    axis.grid(alpha=0.2)

    # (d) Incremental benefit of each paper-facing design.
    axis = axes[1, 1]
    reductions = (
        100 * (1 - float(local_metrics["shape_corrected"]["mape"]) / float(local_metrics["roofline"]["mape"])),
        100 * (1 - float(speed_metrics["policy_calibrated"]["mape"]) / float(speed_metrics["shape_corrected"]["mape"])),
        100 * (1 - float(quality_metrics["structure_aware"]["mae"]) / float(quality_metrics["local_sum"]["mae"])),
        100 * (1 - float(quality_metrics["full_phase_aware"]["mae"]) / float(quality_metrics["structure_aware"]["mae"])),
    )
    labels = ("Shape\ncorrection", "E2E\ncalibration", "Structure-aware\nquality", "Phase-aware\nquality")
    bars = axis.bar(range(4), reductions,
                    color=(COLORS["middle"], COLORS["full"], COLORS["middle"], COLORS["full"]), width=0.68)
    axis.set_xticks(range(4), labels)
    axis.set_ylabel("Relative prediction-error reduction (%)")
    axis.set_title("(d) Incremental design benefit")
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, reductions):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Llama3.1-8B-Instruct modeling ablation", fontsize=14)
    fig.tight_layout()
    fig.savefig(report / "modeling_ablation_detailed.png", dpi=240, bbox_inches="tight")
    fig.savefig(report / "modeling_ablation_detailed.pdf", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "**Figure: Detailed modeling ablation on Llama3.1-8B-Instruct.** "
        "(a) Shape correction is evaluated on 15 independent Llama3 module profiles after excluding exact lookup shapes. "
        "(b) Policy calibration is fitted on seven policies and reports before/after relative error for each of the five held-out policies; the highlighted line is the mean. "
        "(c) Quality predictions use only the fixed 18-policy holdout split. "
        "(d) Each bar reports the relative error reduction over the immediately preceding variant.\n"
    )
    (report / "detailed_caption.md").write_text(caption)


if __name__ == "__main__":
    main()
