#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEBUG_ROOT = Path(__file__).resolve().parents[1]
SOURCE_012 = DEBUG_ROOT.parent / "012_llama2_dialogsum_pareto"


def main() -> None:
    out_dir = DEBUG_ROOT / "summary" / "targeted_dialogsum_calibrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    original = load_original()
    targeted = load_targeted()
    rows = original + targeted
    write_csv(out_dir / "combined_dialogsum_summary.csv", rows)
    plot_global(rows, "conditional_nll", lower_better=True, path=out_dir / "speedup_vs_nll_global.png")
    plot_global(rows, "rougeL", lower_better=False, path=out_dir / "speedup_vs_rougeL_global.png")
    plot_zoom(rows, "conditional_nll", lower_better=True, path=out_dir / "speedup_vs_nll_zoom.png")
    plot_zoom(rows, "rougeL", lower_better=False, path=out_dir / "speedup_vs_rougeL_zoom.png")
    write_report(out_dir / "README.md", rows, targeted)
    print(f"wrote {out_dir}")


def load_original() -> list[dict[str, Any]]:
    path = SOURCE_012 / "summary" / "dialogsum_pareto" / "dialogsum_pareto_summary.csv"
    rows = []
    for row in read_csv(path):
        rows.append(
            {
                "source": "012_original",
                "label": row["label"],
                "kind": row["kind"],
                "point_index": row.get("point_index", ""),
                "speedup": f(row, "measured_speedup_vs_p0"),
                "e2e_ms": f(row, "measured_e2e_ms"),
                "conditional_nll": f(row, "conditional_nll"),
                "rougeL": f(row, "rougeL"),
                "quality_cost": f(row, "quality_cost", math.nan),
            }
        )
    return rows


def load_targeted() -> list[dict[str, Any]]:
    speed_rows = {
        int(row["point_index"]): row
        for row in read_csv(
            DEBUG_ROOT
            / "validation"
            / "stable_e2e_repeats"
            / "targeted_dialogsum_calibrated_card7"
            / "stable_e2e_repeats_summary.csv"
        )
    }
    q_rows = []
    for path in sorted((DEBUG_ROOT / "quality" / "targeted_full_card765").glob("dialogsum_pareto_*.csv")):
        q_rows.extend(read_csv(path))
    rows = []
    for q in q_rows:
        point = int(q["point_index"])
        speed = speed_rows[point]
        rows.append(
            {
                "source": "013_targeted",
                "label": f"C{point}",
                "kind": "calibrated",
                "point_index": point,
                "speedup": f(speed, "e2e_speedup_vs_point0"),
                "e2e_ms": f(speed, "e2e_total_mean_ms"),
                "conditional_nll": f(q, "conditional_nll"),
                "rougeL": f(q, "rougeL"),
                "quality_cost": f(q, "quality_cost"),
            }
        )
    return rows


def plot_global(rows: list[dict[str, Any]], metric: str, *, lower_better: bool, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "pareto": ("o", "#1f77b4", 70),
        "uniform": ("s", "#777777", 70),
        "calibrated": ("*", "#d62728", 180),
    }
    for kind, (marker, color, size) in styles.items():
        items = [row for row in rows if row["kind"] == kind]
        if not items:
            continue
        ax.scatter([row["speedup"] for row in items], [row[metric] for row in items], label=kind, marker=marker, s=size, color=color, alpha=0.9)
    for row in rows:
        if should_label_global(row):
            annotate(ax, row, metric)
    ax.set_xlabel("Measured speedup vs P0")
    ax.set_ylabel(pretty_metric(metric))
    ax.set_title(f"Global view: speedup vs {pretty_metric(metric)}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    add_better_note(ax, lower_better)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_zoom(rows: list[dict[str, Any]], metric: str, *, lower_better: bool, path: Path) -> None:
    zoom_rows = [
        row
        for row in rows
        if row["speedup"] >= 0.98 and not row["label"].startswith("sparse_") and row["label"] != "dense_nvfp4"
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    pareto = [row for row in zoom_rows if row["kind"] == "pareto"]
    pareto.sort(key=lambda row: row["speedup"])
    ax.plot([row["speedup"] for row in pareto], [row[metric] for row in pareto], color="#1f77b4", alpha=0.45, linewidth=1.5)
    styles = {
        "pareto": ("o", "#1f77b4", 80),
        "uniform": ("s", "#555555", 90),
        "calibrated": ("*", "#d62728", 220),
    }
    for kind, (marker, color, size) in styles.items():
        items = [row for row in zoom_rows if row["kind"] == kind]
        ax.scatter([row["speedup"] for row in items], [row[metric] for row in items], label=kind, marker=marker, s=size, color=color, alpha=0.95, zorder=3)
        for row in items:
            annotate(ax, row, metric)
    ax.set_xlabel("Measured speedup vs P0")
    ax.set_ylabel(pretty_metric(metric))
    ax.set_title(f"Zoomed high-speed region: speedup vs {pretty_metric(metric)}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    add_better_note(ax, lower_better)
    pad_axes(ax, zoom_rows, metric)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def should_label_global(row: dict[str, Any]) -> bool:
    if row["kind"] == "calibrated":
        return True
    return row["label"] in {
        "P0",
        "P8",
        "P9",
        "dense_bf16",
        "sparse_bf16",
        "sparse_nvfp4",
        "marlin_nvfp4",
        "dense_nvfp4_prefill_marlin_decode",
    }


def annotate(ax: Any, row: dict[str, Any], metric: str) -> None:
    offsets = {
        "P0": (5, 8),
        "P4": (4, 9),
        "P6": (4, 9),
        "P7": (4, -13),
        "P8": (4, -14),
        "P9": (5, 8),
        "dense_bf16": (4, -14),
        "marlin_nvfp4": (5, 8),
        "dense_nvfp4_prefill_marlin_decode": (-96, -15),
        "C3": (5, 8),
        "C4": (5, 8),
        "C5": (5, -16),
        "sparse_bf16": (5, 8),
        "sparse_nvfp4": (5, 8),
    }
    text = short_label(row["label"])
    ax.annotate(
        text,
        (row["speedup"], row[metric]),
        textcoords="offset points",
        xytext=offsets.get(row["label"], (4, 4)),
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
    )


def short_label(label: str) -> str:
    return {
        "dense_bf16": "dense",
        "dense_nvfp4": "dense nvfp4",
        "sparse_bf16": "sparse bf16",
        "sparse_nvfp4": "sparse nvfp4",
        "marlin_nvfp4": "marlin",
        "dense_nvfp4_prefill_marlin_decode": "hybrid",
    }.get(label, label)


def pretty_metric(metric: str) -> str:
    return {"conditional_nll": "DialogSum NLL", "rougeL": "DialogSum ROUGE-L"}.get(metric, metric)


def add_better_note(ax: Any, lower_better: bool) -> None:
    note = "lower is better" if lower_better else "higher is better"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=9, bbox={"fc": "white", "ec": "#dddddd", "alpha": 0.75})


def pad_axes(ax: Any, rows: list[dict[str, Any]], metric: str) -> None:
    xs = [row["speedup"] for row in rows]
    ys = [row[metric] for row in rows]
    x_pad = max((max(xs) - min(xs)) * 0.08, 0.005)
    y_pad = max((max(ys) - min(ys)) * 0.18, 0.0002)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)


def write_report(path: Path, rows: list[dict[str, Any]], targeted: list[dict[str, Any]]) -> None:
    nll_front = non_dominated(rows, "conditional_nll", lower_better=True)
    rouge_front = non_dominated(rows, "rougeL", lower_better=False)
    lines = [
        "# Targeted DialogSum-Calibrated Pareto",
        "",
        "This experiment recalibrates the original per-module quality proxy with full DialogSum uniform results, then searches only around the high-speed `normal02` region. The calibrated candidates C3/C4/C5 are new policies generated in this 013 experiment; they are not part of the original P0-P9 curve.",
        "",
        "Naming note: C3/C4/C5 mean calibrated candidate point 3/4/5 from the 013 targeted search. I renamed them from the earlier T3/T4/T5 wording to avoid implying that there was a pre-existing `target point` concept.",
        "",
        "## Calibrated Candidate Results",
        "",
        "| label | speedup vs P0 | E2E ms | NLL | ROUGE-L | calibrated cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(targeted, key=lambda r: r["point_index"]):
        lines.append(
            f"| {row['label']} | {row['speedup']:.4f} | {row['e2e_ms']:.2f} | "
            f"{row['conditional_nll']:.6f} | {row['rougeL']:.6f} | {row['quality_cost']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Non-Dominated Rows",
            "",
            f"- Speedup vs NLL: {', '.join(row['label'] for row in nll_front)}.",
            f"- Speedup vs ROUGE-L: {', '.join(row['label'] for row in rouge_front)}.",
            "",
            "## Interpretation",
            "",
            "C4 is the useful point from this pass: it reaches 1.176x speedup with NLL 1.468718 and ROUGE-L 0.160942. It is close to original P8 on NLL while giving better ROUGE-L, but it is slower than original P9 and the uniform hybrid endpoint.",
            "",
            "C5 is close to the marlin/hybrid speed region, but its ROUGE-L is lower than the original high-speed points. It should not be treated as an improved endpoint.",
            "",
            "This confirms the method is moving in the right direction for mid/high-speed tradeoffs, but the calibrated proxy still does not produce a curve that dominates the uniform hybrid endpoint.",
            "",
            "## Next Step",
            "",
            "The next optimization pass should use a latency-constrained formulation with the measured hybrid endpoint as an explicit target, and add a minimum ROUGE/NLL guardrail from T4/P8/P9 rather than relying only on the calibrated aggregate budget.",
            "",
            "## Plots",
            "",
            "The zoomed plots are the main plots to inspect. The global plots are mainly sanity checks because sparse uniform points stretch the y-axis and make the useful high-speed region hard to read.",
            "",
            "![Zoomed Speedup vs NLL](speedup_vs_nll_zoom.png)",
            "",
            "![Zoomed Speedup vs ROUGE-L](speedup_vs_rougeL_zoom.png)",
            "",
            "![Global Speedup vs NLL](speedup_vs_nll_global.png)",
            "",
            "![Global Speedup vs ROUGE-L](speedup_vs_rougeL_global.png)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def non_dominated(rows: list[dict[str, Any]], metric: str, *, lower_better: bool) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            speed_ok = other["speedup"] >= row["speedup"]
            metric_ok = other[metric] <= row[metric] if lower_better else other[metric] >= row[metric]
            strict = other["speedup"] > row["speedup"] or (other[metric] < row[metric] if lower_better else other[metric] > row[metric])
            if speed_ok and metric_ok and strict:
                dominated = True
                break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda r: r["speedup"])


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
