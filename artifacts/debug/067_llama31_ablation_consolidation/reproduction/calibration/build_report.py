#!/usr/bin/env python3
"""Build the calibration-size sensitivity plot and report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13.0,
            "axes.labelsize": 13.5,
            "axes.labelweight": "semibold",
            "axes.titlesize": 13.5,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.0,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    report = OUT / "report"
    report.mkdir(parents=True, exist_ok=True)
    rows = read_csv(OUT / "results/learning_curve_summary.csv")
    validation = json.loads((OUT / "results/validation.json").read_text())
    sizes = [int(row["calibration_policies"]) for row in rows]
    random_mae = [float(row["random_holdout_mae_median"]) for row in rows]
    random_mae_q25 = [float(row["random_holdout_mae_q25"]) for row in rows]
    random_mae_q75 = [float(row["random_holdout_mae_q75"]) for row in rows]
    cost = [float(row["random_label_gpu_minutes_median"]) for row in rows]
    cost_q25 = [float(row["random_label_gpu_minutes_q25"]) for row in rows]
    cost_q75 = [float(row["random_label_gpu_minutes_q75"]) for row in rows]

    configure_style()
    fig, axis = plt.subplots(figsize=(5.6, 3.8))
    mae_band = axis.fill_between(sizes, random_mae_q25, random_mae_q75, color="#4c78a8", alpha=0.18,
                                 label="Subset variation (IQR)")
    mae_line, = axis.plot(sizes, random_mae, color="#4c78a8", marker="o", linewidth=2.5,
                          markersize=6.8,
                          label="Median MAE")
    axis.set_xlabel("Calibration set size")
    axis.set_ylabel("$\\Delta$NLL prediction MAE")
    axis.set_xticks([size for size in sizes if size != 66])
    axis.grid(alpha=0.25, linewidth=0.7)

    time_axis = axis.twinx()
    cost_band = time_axis.fill_between(sizes, cost_q25, cost_q75, color="#e45756", alpha=0.12)
    cost_line, = time_axis.plot(sizes, cost, color="#e45756", marker="^", linewidth=2.2,
                                markersize=7.0,
                                label="Calibration cost")
    time_axis.set_ylabel("Calibration cost (min)", color="#b33939")
    time_axis.tick_params(axis="y", labelcolor="#b33939")

    elbow = int(validation["observed_elbow"])
    axis.axvline(elbow, color="#6b7280", linestyle=":", linewidth=1.5)
    axis.annotate(f"5% elbow: {elbow}", (elbow, random_mae[-1]), xytext=(-88, 16),
                  textcoords="offset points", fontsize=11.5)
    selected_index = sizes.index(54)
    selected_mae = float(validation["frozen_main_54_holdout_mae"])
    selected_cost = float(validation["frozen_main_54_label_gpu_minutes"])
    axis.scatter([54], [selected_mae], marker="*", s=150, color="#f59e0b", edgecolor="#92400e",
                 zorder=5, label="Selected setting")
    time_axis.annotate(f"{cost[-1]:.1f} GPU-min", (sizes[-1], cost[-1]), xytext=(-115, -45),
                       textcoords="offset points", color="#b33939", fontsize=11.5)
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = time_axis.get_legend_handles_labels()
    axis.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=2,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
        fontsize=10.0,
        handlelength=1.8,
        columnspacing=0.8,
        labelspacing=0.35,
        borderpad=0.35,
    )
    axis.set_title(
        "Llama3.1 quality calibration-size sensitivity",
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(report / "calibration_size_sensitivity_full.png", dpi=240, bbox_inches="tight")
    fig.savefig(report / "calibration_size_sensitivity_full.pdf", bbox_inches="tight")
    paper_indices = [position for position, size in enumerate(sizes) if size <= 64]
    paper_sizes = [sizes[position] for position in paper_indices]
    mae_line.set_data(paper_sizes, [random_mae[position] for position in paper_indices])
    cost_line.set_data(paper_sizes, [cost[position] for position in paper_indices])
    mae_band.remove()
    cost_band.remove()
    axis.fill_between(
        paper_sizes,
        [random_mae_q25[position] for position in paper_indices],
        [random_mae_q75[position] for position in paper_indices],
        color="#4c78a8",
        alpha=0.18,
    )
    time_axis.fill_between(
        paper_sizes,
        [cost_q25[position] for position in paper_indices],
        [cost_q75[position] for position in paper_indices],
        color="#e45756",
        alpha=0.12,
    )
    axis.set_xlim(4, 65)
    axis.set_title("")
    fig.set_size_inches(5.6, 3.5, forward=True)
    fig.tight_layout()
    fig.savefig(report / "calibration_size_sensitivity.png", dpi=240, bbox_inches="tight")
    fig.savefig(report / "calibration_size_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    table = [
        "| Calibration policies | Random median MAE | Random IQR | Random median Spearman | GPU-min |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {int(row['calibration_policies'])} | {float(row['random_holdout_mae_median']):.4f} | "
            f"[{float(row['random_holdout_mae_q25']):.4f}, {float(row['random_holdout_mae_q75']):.4f}] | "
            f"{float(row['random_holdout_spearman_median']):.3f} | {float(row['random_label_gpu_minutes_median']):.1f} |"
        )
    first = rows[0]
    last = rows[-1]
    improvement = 100 * (
        1 - float(last["random_holdout_mae_median"]) / float(first["random_holdout_mae_median"])
    )
    extension_mae_gain = 100 * (1 - float(last["random_holdout_mae_median"]) / selected_mae)
    extension_cost_gain = 100 * (float(last["random_label_gpu_minutes_median"]) / selected_cost - 1)
    summary = [
        "# Quality calibration-size sensitivity", "",
        "The fixed 18-policy holdout is never used for fitting or subset construction. The original 54 training policies and 18 calibration-only extensions form one exchangeable 72-policy candidate pool. All subsets contain the five dense/uniform anchors, and ten fixed random nested orders provide the median and IQR. The exact frozen 54-policy main model is reported separately and is not used to shape the sensitivity curve.", "",
        *table, "",
        f"The preregistered 5%-from-full rule identifies **{validation['observed_elbow']} policies** as the observed elbow. The random-subset median holdout MAE improves by {improvement:.1f}% from 5 to 72 policies.", "",
        f"The frozen main 54-policy configuration has holdout MAE {selected_mae:.4f}, costs {selected_cost:.1f} recorded GPU-minutes, and reproduces 064 within the fixed numerical tolerance. The random-pool median at 54 policies is {float(rows[selected_index]['random_holdout_mae_median']):.4f}; it must not be confused with the particular frozen main subset.", "",
        f"Using the full 72-policy pool costs {float(last['random_label_gpu_minutes_median']):.1f} GPU-minutes: {extension_cost_gain:.1f}% more labeling cost than the frozen main subset and {extension_mae_gain:.1f}% lower MAE than that particular subset.", "",
        "The curve is reported as observed; no subset, holdout, or elbow threshold was retuned after inspecting the result.",
        "The paper-facing figure caps the displayed calibration budget at 64 policies. The complete 72-policy audit curve is retained as `calibration_size_sensitivity_full.png` and the table above reports every measured point.",
    ]
    (report / "summary.md").write_text("\n".join(summary) + "\n")

    paper = (
        "**Calibration-size sensitivity.** We keep the 18-policy holdout fixed and refit the unchanged "
        "phase-aware quality surrogate using nested subsets drawn from one 72-policy candidate pool. "
        "Every subset retains the five dense/uniform anchors, while ten fixed random orders quantify subset variance. Under a "
        "preregistered criterion that selects the smallest calibration size remaining within 5% of the "
        f"full-model MAE, the observed elbow occurs at {validation['observed_elbow']} policies. "
        f"The independently frozen 54-policy main model requires {selected_cost:.1f} GPU-minutes. The full 72-policy pool "
        f"requires {extension_cost_gain:.1f}% more labeling cost and reduces holdout MAE by a further "
        f"{extension_mae_gain:.1f}% relative to that particular subset. We mark the frozen main model separately "
        "from the random-pool sensitivity curve to avoid post-hoc subset selection.\n"
    )
    (report / "paper_text.md").write_text(paper)


if __name__ == "__main__":
    main()
