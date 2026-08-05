#!/usr/bin/env python3
"""Build the compact paper figure and textual report for the 064 ablation."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parents[1]
QUALITY_NAMES = {
    "local_sum": "Local sum",
    "structure_aware": "+ Structure-aware",
    "full_phase_aware": "+ Phase-aware",
}
SPEED_NAMES = {
    "roofline": "Roofline",
    "shape_corrected": "+ Shape correction",
    "roofline_policy_calibrated": "Roofline + affine E2E calibration",
    "policy_calibrated": "+ Shape correction + affine E2E calibration",
}
COLORS = ("#9ca3af", "#4c78a8", "#e45756")


def read_holdout(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "holdout"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = OUT / "report"
    report.mkdir(parents=True, exist_ok=True)
    speed = read_holdout(OUT / "speed/metrics.csv")
    local_speed = read_rows(OUT / "speed/local_metrics.csv")
    kernel_generalization = read_rows(OUT / "speed/kernel_generalization_metrics.csv")
    quality = read_holdout(OUT / "quality/metrics.csv")
    speed_by_name = {row["variant"]: row for row in speed}
    local_speed_by_name = {row["variant"]: row for row in local_speed}
    kernel_generalization_by_name = {
        row["variant"]: row for row in kernel_generalization if row["kernel"] == "all"
    }
    quality_by_name = {row["variant"]: row for row in quality}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    speed_positions = (0.0, 1.0, 2.6, 3.6)
    speed_labels = ("Roofline", "+ Shape correction", "Raw policy sum", "+ E2E calibration")
    speed_values = (
        100 * float(local_speed_by_name["roofline"]["mape"]),
        100 * float(local_speed_by_name["shape_corrected"]["mape"]),
        100 * float(speed_by_name["shape_corrected"]["mape"]),
        100 * float(speed_by_name["policy_calibrated_pava"]["mape"]),
    )
    bars = axes[0].bar(speed_positions, speed_values, color=(COLORS[0], COLORS[1], COLORS[0], COLORS[2]), width=0.68)
    axes[0].set_xticks(speed_positions, speed_labels, rotation=12, ha="right")
    axes[0].set_ylabel("Latency MAPE (%)")
    axes[0].set_title("(a) Latency model")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].axvline(1.8, color="#9ca3af", linestyle="--", linewidth=1)
    axes[0].text(0.5, -0.28, "Kernel-level", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=9)
    axes[0].text(3.1, -0.28, "Policy-level", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=9)
    for bar, value in zip(bars, speed_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    quality_order = list(QUALITY_NAMES)
    quality_values = [float(quality_by_name[name]["mae"]) for name in quality_order]
    bars = axes[1].bar(range(3), quality_values, color=COLORS, width=0.68)
    axes[1].set_xticks(range(3), [QUALITY_NAMES[name] for name in quality_order], rotation=12, ha="right")
    axes[1].set_ylabel("Held-out ΔNLL MAE")
    axes[1].set_title("(b) Quality model")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, name in zip(bars, quality_order):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"ρ={float(quality_by_name[name]['spearman']):.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Llama3.1-8B-Instruct, prefill-decode modeling ablation", fontsize=13)
    fig.tight_layout()
    fig.savefig(report / "modeling_ablation.png", dpi=240, bbox_inches="tight")
    fig.savefig(report / "modeling_ablation.pdf", bbox_inches="tight")
    plt.close(fig)

    local_speed_lines = ["| Variant | Samples | MAE (ms) | MAPE | Spearman |", "|---|---:|---:|---:|---:|"]
    for name in ("roofline", "shape_corrected"):
        row = local_speed_by_name[name]
        local_speed_lines.append(
            f"| {SPEED_NAMES[name]} | {row['samples']} | {float(row['mae_ms']):.4f} | {100*float(row['mape']):.2f}% | {float(row['spearman']):.3f} |"
        )
    kernel_lines = ["| Variant | Samples | MAPE | p90 APE | Spearman |", "|---|---:|---:|---:|---:|"]
    kernel_names = {
        "direct_fit": "Direct shape fit",
        "roofline": "+ Calibrated Roofline prior",
        "roofline_shape_corrected": "+ Shape residual correction",
    }
    for name in kernel_names:
        row = kernel_generalization_by_name[name]
        kernel_lines.append(
            f"| {kernel_names[name]} | {row['samples']} | {100*float(row['mape']):.2f}% | {100*float(row['p90_ape']):.2f}% | {float(row['spearman']):.3f} |"
        )
    speed_lines = ["| Variant | MAE (ms) | MAPE | Spearman |", "|---|---:|---:|---:|"]
    speed_display = {
        "shape_corrected": "Raw E2E aggregation",
        "policy_calibrated_pava": "+ Monotonic E2E calibration",
    }
    for name, display in speed_display.items():
        row = speed_by_name[name]
        speed_lines.append(
            f"| {display} | {float(row['mae_ms']):.2f} | {100*float(row['mape']):.2f}% | {float(row['spearman']):.3f} |"
        )
    quality_lines = ["| Variant | MAE | RMSE | Spearman |", "|---|---:|---:|---:|"]
    for name in quality_order:
        row = quality_by_name[name]
        quality_lines.append(
            f"| {QUALITY_NAMES[name]} | {float(row['mae']):.4f} | {float(row['rmse']):.4f} | {float(row['spearman']):.3f} |"
        )

    local_first, local_last = local_speed_by_name["roofline"], local_speed_by_name["shape_corrected"]
    speed_first, speed_last = speed_by_name["shape_corrected"], speed_by_name["policy_calibrated_pava"]
    quality_first, quality_last = quality_by_name[quality_order[0]], quality_by_name[quality_order[-1]]
    local_reduction = 100 * (1 - float(local_last["mape"]) / float(local_first["mape"]))
    speed_reduction = 100 * (1 - float(speed_last["mape"]) / float(speed_first["mape"]))
    quality_reduction = 100 * (1 - float(quality_last["mae"]) / float(quality_first["mae"]))
    summary = [
        "# Llama3.1 modeling ablation", "",
        "The experiment uses only frozen RTX 5090 measurements from 057: 54/18 quality policies and 7/5 speed policies.", "",
        "## Latency model", "", "### Strict unseen-(N,K) kernel generalization", "", *kernel_lines, "",
        "All three variants use the same shape features, ridge strength, local-neighbor mixture, seeds, and 25% held-out (N,K) groups. The direct-fit baseline predicts log latency; the other variants add the method-aware Roofline prior and its residual correction.", "",
        "### Independent Llama3 module profiles", "", *local_speed_lines, "",
        f"Shape correction reduces local MAPE by {local_reduction:.1f}% relative to the method-aware Roofline baseline.", "",
        "### Held-out heterogeneous policies", "", *speed_lines, "",
        f"With the refined kernel predictor fixed, the PAVA monotonic E2E calibration reduces policy-level MAPE by {speed_reduction:.1f}%, from {100*float(speed_first['mape']):.2f}% to {100*float(speed_last['mape']):.2f}%.", "",
        "## Quality model", "", *quality_lines, "",
        f"The full quality model reduces held-out MAE by {quality_reduction:.1f}% relative to direct local-error summation.", "",
        "## Interpretation", "",
        "The rows are fixed before inspecting holdout labels. Non-monotonic intermediate results, if any, are retained rather than hidden or retuned on the holdout set.",
    ]
    (report / "summary.md").write_text("\n".join(summary) + "\n")

    paper = [
        "# Draft-ready ablation text", "",
        "**Latency-model ablation.** On strictly unseen (N,K) groups, direct shape fitting, a calibrated Roofline prior, and Roofline plus shape-residual correction obtain MAPE values of "
        f"{100*float(kernel_generalization_by_name['direct_fit']['mape']):.2f}%, {100*float(kernel_generalization_by_name['roofline']['mape']):.2f}%, and {100*float(kernel_generalization_by_name['roofline_shape_corrected']['mape']):.2f}%, respectively. "
        f"With the refined kernel predictor fixed, PAVA monotonic E2E calibration reduces policy-level MAPE from {100*float(speed_first['mape']):.2f}% to {100*float(speed_last['mape']):.2f}%.", "",
        "**Quality-model ablation.** Directly summing local perturbations gives a held-out ΔNLL MAE of "
        f"{float(quality_first['mae']):.4f}. Structure-aware weighting changes it to {float(quality_by_name['structure_aware']['mae']):.4f}, "
        f"and the complete phase-aware nonnegative factorization reaches {float(quality_last['mae']):.4f} with a Spearman correlation of {float(quality_last['spearman']):.3f}.",
    ]
    (report / "paper_text.md").write_text("\n".join(paper) + "\n")

    table = [
        "# Compact ablation tables", "",
        "## Latency prediction", "", *kernel_lines, "", *speed_lines, "",
        "## Quality prediction", "", *quality_lines, "",
        "Kernel rows report MAPE on unseen (N,K) benchmark groups; policy rows report MAPE on five held-out heterogeneous policies; quality rows report ΔNLL MAE on 18 held-out policies.",
    ]
    (report / "ablation_tables.md").write_text("\n".join(table) + "\n")

    tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Modeling ablation on Llama3.1-8B-Instruct.}",
        r"\begin{tabular}{ccc}", r"\toprule", r"Ablation type & Variant & Error $\downarrow$ \\", r"\midrule",
    ]
    kernel_tex_names = {
        "direct_fit": "Direct fit",
        "roofline": "+ Roofline prior",
        "roofline_shape_corrected": "+ Shape correction",
    }
    for position, name in enumerate(kernel_names):
        row = kernel_generalization_by_name[name]
        ablation_type = r"\multirow{3}{*}{Kernel prediction}" if position == 0 else ""
        tex.append(f"{ablation_type} & {kernel_tex_names[name]} & {100*float(row['mape']):.2f}\\% \\\\")
    tex.append(r"\midrule")
    e2e_names = {
        "shape_corrected": "Raw E2E aggregation",
        "policy_calibrated_pava": "+ Monotonic E2E calibration",
    }
    for position, name in enumerate(e2e_names):
        row = speed_by_name[name]
        ablation_type = r"\multirow{2}{*}{E2E composition}" if position == 0 else ""
        tex.append(f"{ablation_type} & {e2e_names[name]} & {100*float(row['mape']):.2f}\\% \\\\")
    tex.append(r"\midrule")
    quality_tex_names = {
        "local_sum": "Local aggregation",
        "structure_aware": "+ Structure awareness",
        "full_phase_aware": "+ Phase awareness",
    }
    for position, name in enumerate(quality_order):
        row = quality_by_name[name]
        ablation_type = r"\multirow{3}{*}{Quality prediction}" if position == 0 else ""
        tex.append(f"{ablation_type} & {quality_tex_names[name]} & {float(row['mae']):.4f} \\\\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\label{tab:modeling-ablation}", r"\end{table}"])
    (report / "ablation_tables.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
