#!/usr/bin/env python3
"""Build matched Llama3.1 paper figures for both inference scenarios."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/068_llama31_prefill_uniform_coverage_refinement"
PREFILL = EXP / "llama31_8b_instruct/report/all_results.csv"
DECODE = (
    ROOT
    / "artifacts/debug/060_two_model_two_scenario_result_consolidation"
    / "llama31_8b_instruct/prefill_decode/data/complete_results.csv"
)
OUT = EXP / "llama31_8b_instruct/report/paper"

SINGLE_COLUMN_WIDTH = 3.35
SINGLE_FIGURE_HEIGHT = 2.45
STACKED_FIGURE_HEIGHT = 4.45
HORIZONTAL_FIGURE_HEIGHT = 2.05
PNG_DPI = 300

UNIFORM_LABELS = {
    "dense_bf16": "BF16",
    "dense_nvfp4": "NVFP4",
    "sparse_bf16": "2:4 BF16",
    "sparse_nvfp4": "2:4 NVFP4",
    "w4a16_ours": "W4A16",
    "p00": "BF16",
    "p01": "NVFP4",
    "p02": "2:4 BF16",
    "p03": "2:4 NVFP4",
    "p04": "W4A16",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the exact non-dominated set when speed and quality are maximized."""
    retained = []
    for row in rows:
        dominated = any(
            float(other["speedup"]) >= float(row["speedup"])
            and float(other["quality"]) >= float(row["quality"])
            and (
                float(other["speedup"]) > float(row["speedup"])
                or float(other["quality"]) > float(row["quality"])
            )
            for other in rows
            if other is not row
        )
        if not dominated:
            retained.append(row)
    return sorted(retained, key=lambda row: float(row["speedup"]))


def load_prefill() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ours, uniform = [], []
    for row in read(PREFILL):
        if not row["speedup"] or not row["arc_challenge"]:
            continue
        item = {
            "policy": row["policy"],
            "speedup": float(row["speedup"]),
            "quality": float(row["arc_challenge"]),
        }
        (uniform if row["family"] == "uniform" else ours).append(item)
    return ours, uniform


def load_decode(metric: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ours, uniform = [], []
    for row in read(DECODE):
        policy = row["policy_id"]
        if not row["measured_speedup_vs_dense"] or not row[metric]:
            continue
        item = {
            "policy": policy,
            "speedup": float(row["measured_speedup_vs_dense"]),
            "quality": float(row[metric]),
        }
        if policy.startswith("point_"):
            # The source report marks point_007's 0.739x speed as anomalous.
            # Retain it in the source CSV but exclude it from the paper plot.
            if policy != "point_007":
                ours.append(item)
        elif policy in {"p00", "p01", "p02", "p03", "p04"}:
            uniform.append(item)
    return ours, uniform


def coverage(
    scenario: str,
    ours: list[dict[str, object]],
    uniform: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for reference in uniform:
        dominators = [
            row
            for row in ours
            if float(row["speedup"]) >= float(reference["speedup"])
            and float(row["quality"]) >= float(reference["quality"])
        ]
        rows.append(
            {
                "scenario": scenario,
                "uniform": reference["policy"],
                "uniform_speedup": reference["speedup"],
                "uniform_quality": reference["quality"],
                "covered": bool(dominators),
                "example_ours": (
                    max(dominators, key=lambda row: float(row["speedup"]))["policy"]
                    if dominators
                    else ""
                ),
            }
        )
    return rows


def draw(
    ax,
    ours: list[dict[str, object]],
    uniform: list[dict[str, object]],
    ylabel: str,
    panel_title: str | None,
    annotations: dict[str, tuple[int, int, str]],
    annotation_fontsize: float = 6.5,
    uniform_annotation_offsets: dict[str, tuple[int, int]] | None = None,
) -> None:
    edge = frontier(ours)
    ax.plot(
        [row["speedup"] for row in edge],
        [row["quality"] for row in edge],
        color="#1f2937",
        marker="o",
        markeredgewidth=0.4,
        markersize=4.0,
        linewidth=1.35,
        label="Ours frontier",
        zorder=3,
    )
    ax.scatter(
        [row["speedup"] for row in uniform],
        [row["quality"] for row in uniform],
        marker="s",
        s=25,
        color="#dc2626",
        linewidths=0.4,
        label="Uniform",
        zorder=4,
    )

    for row in uniform:
        policy = str(row["policy"])
        dx, dy = (4, 3)
        if policy in {"dense_bf16", "w4a16_ours", "p00", "p04", "dense_nvfp4"}:
            dx, dy = (4, -10)
        if uniform_annotation_offsets and policy in uniform_annotation_offsets:
            dx, dy = uniform_annotation_offsets[policy]
        ax.annotate(
            UNIFORM_LABELS[policy],
            (float(row["speedup"]), float(row["quality"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=annotation_fontsize,
            color="#b91c1c",
            annotation_clip=True,
        )

    for row in edge:
        policy = str(row["policy"])
        if policy not in annotations:
            continue
        dx, dy, label = annotations[policy]
        ax.annotate(
            label,
            (float(row["speedup"]), float(row["quality"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=annotation_fontsize,
            color="#111827",
            annotation_clip=True,
        )

    if panel_title:
        ax.set_title(panel_title, pad=3)
    ax.set_xlabel("Measured E2E speedup vs. BF16", labelpad=2)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#9ca3af", linewidth=0.45, alpha=0.28)
    ax.margins(x=0.045, y=0.055)


def save_single(
    name: str,
    ours: list[dict[str, object]],
    uniform: list[dict[str, object]],
    ylabel: str,
    annotations: dict[str, tuple[int, int, str]],
) -> None:
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, SINGLE_FIGURE_HEIGHT),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.025, h_pad=0.025)
    draw(ax, ours, uniform, ylabel, None, annotations)
    ax.legend(
        frameon=True,
        loc="best",
        borderpad=0.35,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.3,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=PNG_DPI)
    plt.close(fig)


def save_horizontal(
    prefill_ours: list[dict[str, object]],
    prefill_uniform: list[dict[str, object]],
    decode_ours: list[dict[str, object]],
    decode_uniform: list[dict[str, object]],
) -> None:
    """Save a side-by-side version that still fits a single paper column."""
    with plt.rc_context(
        {
            "font.size": 6.5,
            "axes.labelsize": 6.7,
            "axes.titlesize": 7.1,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(SINGLE_COLUMN_WIDTH, HORIZONTAL_FIGURE_HEIGHT),
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.015, h_pad=0.018, wspace=0.04)
        draw(
            axes[0],
            prefill_ours,
            prefill_uniform,
            "ARC-C acc_norm",
            "(a) Prefill-only",
            {},
            annotation_fontsize=5.4,
        )
        draw(
            axes[1],
            decode_ours,
            decode_uniform,
            "IWSLT SacreBLEU",
            "(b) Prefill-decode",
            {},
            annotation_fontsize=5.4,
            uniform_annotation_offsets={"p00": (-2, -12), "p04": (7, -19)},
        )
        highlight_specs = (
            ("Balanced", "D", "#0072B2", 18),
            ("Max speed", "*", "#E69F00", 38),
        )
        for ax, rows, policies in (
            (axes[0], prefill_ours, ("fine_q208", "point_014")),
            (axes[1], decode_ours, ("point_005", "point_011")),
        ):
            for policy, (label, marker, color, size) in zip(
                policies, highlight_specs, strict=True
            ):
                row = next(item for item in rows if item["policy"] == policy)
                ax.scatter(
                    [row["speedup"]],
                    [row["quality"]],
                    marker=marker,
                    s=size,
                    color=color,
                    edgecolors="white",
                    linewidths=0.4,
                    label=label,
                    zorder=5,
                )
        for ax in axes:
            ax.set_xlabel("")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        fig.supxlabel("Measured E2E speedup vs. BF16", fontsize=6.7)
        handles, labels = axes[0].get_legend_handles_labels()
        labels = ["Frontier" if label == "Ours frontier" else label for label in labels]
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=4,
            frameon=True,
            borderpad=0.25,
            handlelength=1.35,
            handletextpad=0.25,
            columnspacing=0.45,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(
                OUT / f"llama31_two_scenario_tradeoff_horizontal.{suffix}",
                dpi=PNG_DPI,
            )
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prefill_ours, prefill_uniform = load_prefill()
    decode_ours, decode_uniform = load_decode("iwslt_sacre_bleu")
    cnn_ours, cnn_uniform = load_decode("cnn_dm_rougeL_percent")

    prefill_annotations = {
        "fine_q208": (4, 5, "Balanced"),
        "point_014": (-41, -10, "Max speed"),
    }
    decode_annotations = {
        "point_005": (4, 5, "Balanced"),
        "point_011": (-41, -10, "Max speed"),
    }
    save_single(
        "prefill_only_arc_challenge",
        prefill_ours,
        prefill_uniform,
        "ARC-Challenge acc_norm",
        prefill_annotations,
    )
    save_single(
        "prefill_decode_iwslt_bleu",
        decode_ours,
        decode_uniform,
        "IWSLT SacreBLEU",
        decode_annotations,
    )
    # Retain CNN/DM as a consistently styled alternative for an appendix.
    save_single(
        "prefill_decode_cnn_dm_rougel",
        cnn_ours,
        cnn_uniform,
        "CNN/DM ROUGE-L (%)",
        decode_annotations,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(SINGLE_COLUMN_WIDTH, STACKED_FIGURE_HEIGHT),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.025, h_pad=0.025, hspace=0.06)
    draw(
        axes[0],
        prefill_ours,
        prefill_uniform,
        "ARC-Challenge acc_norm",
        "(a) Prefill-only",
        prefill_annotations,
    )
    draw(
        axes[1],
        decode_ours,
        decode_uniform,
        "IWSLT SacreBLEU",
        "(b) Prefill-decode",
        decode_annotations,
    )
    axes[0].set_xlabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=True,
        borderpad=0.35,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"llama31_two_scenario_tradeoff.{suffix}", dpi=PNG_DPI)
    plt.close(fig)
    save_horizontal(prefill_ours, prefill_uniform, decode_ours, decode_uniform)

    audits = coverage("prefill_only_arc_challenge", prefill_ours, prefill_uniform)
    audits += coverage("prefill_decode_iwslt_bleu", decode_ours, decode_uniform)
    with (OUT / "coverage_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)
    if not all(row["covered"] for row in audits):
        raise RuntimeError("A uniform baseline is outside the measured ours envelope")

    lines = [
        "# Llama3.1-8B-Instruct two-scenario paper trade-off figures",
        "",
        "- Prefill-only: ARC-Challenge `acc_norm`, sourced from debug 068.",
        "- Prefill-decode main panel: IWSLT SacreBLEU, sourced from debug 060/057.",
        "- CNN/DM ROUGE-L is retained as a standalone appendix alternative.",
        "- Both axes use measured E2E speedup and real downstream-task quality.",
        "- Only exact non-dominated ours points are drawn; dominated measurements remain in the source CSV.",
        "- `point_007` remains in the source table but is excluded from the prefill-decode paper plot because its source speed is explicitly marked anomalous.",
        "- Every uniform baseline is programmatically covered by an ours point in both selected plots; see `coverage_audit.csv`.",
        "- All figures are sized for a 3.35-inch paper column; the two-panel figure is stacked vertically.",
        "- A 3.35-inch side-by-side alternative is provided as `llama31_two_scenario_tradeoff_horizontal`.",
        "- In the side-by-side figure, blue diamonds mark Balanced and orange stars mark Max speed.",
        "- Direct labels use compact names: `2:4` means structured sparse and `W4A16` is the Marlin baseline.",
        "",
        "The PNG files are convenient for inspection. Use the corresponding PDF files in the paper.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
