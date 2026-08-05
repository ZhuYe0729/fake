#!/usr/bin/env python3
"""Rebuild Llama2 prefill-decode plots from the final measured tables."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
SCENARIO = (
    ROOT
    / "artifacts/debug/060_two_model_two_scenario_result_consolidation"
    / "llama2_7b_chat/prefill_decode"
)

METRICS = (
    (
        "cnn_dm_rougeL_percent",
        "CNN/DM ROUGE-L (%)",
        "pareto_cnn_dm_1000.png",
        "pareto_cnn_dm_1000_with_uniform_056.png",
    ),
    (
        "dsum_rougeL_percent",
        "DialogSum ROUGE-L (%)",
        "pareto_dsum.png",
        "pareto_dsum_with_uniform_056.png",
    ),
    (
        "iwslt_sacre_bleu",
        "IWSLT SacreBLEU",
        "pareto_IWSLT.png",
        "pareto_IWSLT_with_uniform_056.png",
    ),
)

UNIFORM_LABELS = {
    "p00": "Dense BF16",
    "p01": "Dense NVFP4",
    "p02": "Sparse BF16",
    "p03": "Sparse NVFP4 projection",
    "p04": "W4A16/Marlin",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def measured(row: dict[str, str], field: str) -> float:
    return float(row[field])


def frontier(rows: list[dict[str, str]], metric: str) -> list[dict[str, str]]:
    retained = []
    for row in rows:
        speed = measured(row, "measured_speedup_vs_dense")
        quality = measured(row, metric)
        dominated = any(
            measured(other, "measured_speedup_vs_dense") >= speed
            and measured(other, metric) >= quality
            and (
                measured(other, "measured_speedup_vs_dense") > speed
                or measured(other, metric) > quality
            )
            for other in rows
            if other is not row
        )
        if not dominated:
            retained.append(row)
    return sorted(retained, key=lambda row: measured(row, "measured_speedup_vs_dense"))


def plot_metric(
    ours: list[dict[str, str]],
    uniform: list[dict[str, str]],
    metric: str,
    ylabel: str,
    outputs: tuple[str, str],
) -> None:
    edge = frontier(ours, metric)
    fig, ax = plt.subplots(figsize=(8.4, 5.7))

    ax.scatter(
        [measured(row, "measured_speedup_vs_dense") for row in ours],
        [measured(row, metric) for row in ours],
        s=38,
        color="#1f2937",
        alpha=0.28,
        label="Ours (all measured policies)",
        zorder=2,
    )
    ax.plot(
        [measured(row, "measured_speedup_vs_dense") for row in edge],
        [measured(row, metric) for row in edge],
        color="#1f2937",
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        label="Ours (non-dominated frontier)",
        zorder=3,
    )
    ax.scatter(
        [measured(row, "speedup_vs_dense_bf16") for row in uniform],
        [measured(row, metric) for row in uniform],
        s=82,
        marker="s",
        color="#dc2626",
        label="Uniform baselines",
        zorder=4,
    )

    for index, row in enumerate(uniform):
        x = measured(row, "speedup_vs_dense_bf16")
        y = measured(row, metric)
        dy = 8 if index % 2 == 0 else -15
        ax.annotate(
            UNIFORM_LABELS[row["policy_id"]],
            (x, y),
            xytext=(7, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="#b91c1c",
        )

    ax.set_xlabel("Measured E2E speedup vs dense BF16")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    for output in outputs:
        fig.savefig(SCENARIO / "figures" / output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(
        f"{metric}: {len(ours)} ours, {len(edge)} frontier, "
        f"{len(uniform)} uniform -> {', '.join(outputs)}"
    )


def main() -> None:
    ours = read(SCENARIO / "data/complete_results.csv")
    uniform = read(SCENARIO / "data/uniform_baseline_results.csv")
    if {row["policy_id"] for row in uniform} != set(UNIFORM_LABELS):
        raise RuntimeError("Expected final p00-p04 uniform baseline table")

    for metric, ylabel, output, compatibility_output in METRICS:
        measured_ours = [row for row in ours if row.get(metric)]
        if any(not row.get(metric) for row in uniform):
            raise RuntimeError(f"Missing uniform measurement for {metric}")
        plot_metric(
            measured_ours,
            uniform,
            metric,
            ylabel,
            (output, compatibility_output),
        )


if __name__ == "__main__":
    main()
