#!/usr/bin/env python3
"""Render the measured two-scenario Pareto summary used for paper review."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], name: str) -> float:
    return float(row[name])


def plot(ax, data, title: str) -> None:
    ours = [row for row in data if row["family"] == "ours"]
    refs = [row for row in data if row["family"] != "ours"]
    kept = [row for row in ours if row["globally_pareto_kept"] == "True"]
    ax.plot([value(row, "measured_wikitext_delta_nll") for row in kept],
            [value(row, "e2e_median_ms") for row in kept], color="#0072B2", marker="o",
            linewidth=2, label="Ours (real Pareto points)")
    dominated_ours = [row for row in ours if row["globally_pareto_kept"] != "True"]
    if dominated_ours:
        ax.scatter([value(row, "measured_wikitext_delta_nll") for row in dominated_ours],
                   [value(row, "e2e_median_ms") for row in dominated_ours], marker="x",
                   color="#0072B2", s=45, label="Ours (dominated)")
    for row in refs:
        kept_ref = row["globally_pareto_kept"] == "True"
        ax.scatter(value(row, "measured_wikitext_delta_nll"), value(row, "e2e_median_ms"),
                   marker="s" if kept_ref else "x", color="#D55E00", s=48)
        ax.annotate(row["label"], (value(row, "measured_wikitext_delta_nll"), value(row, "e2e_median_ms")),
                    xytext=(4, 4), textcoords="offset points", fontsize=7, color="#803500")
    ax.set_title(title)
    ax.set_xlabel("WikiText ΔNLL (lower is better)")
    ax.set_ylabel("Median E2E latency (ms, lower is better)")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="best")


def markdown_table(data: list[dict[str, str]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in data:
        rendered = []
        for field in fields:
            cell = row[field]
            if field in {"measured_wikitext_delta_nll", "e2e_median_ms"}:
                cell = f"{float(cell):.4f}"
            rendered.append(cell)
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "report"
    out.mkdir(exist_ok=True)
    prefill = rows(root / "validation" / "prefill_only" / "measured_comparison.csv")
    decode = rows(root / "validation" / "prefill_decode" / "measured_comparison_official.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    plot(axes[0], prefill, "Prefill-only (b=8, input=2048)")
    plot(axes[1], decode, "Prefill-decode (b=16, 2048+80)")
    fig.savefig(out / "measured_pareto.png", dpi=220)
    fields = ["family", "label", "measured_wikitext_delta_nll", "e2e_median_ms", "globally_pareto_kept"]
    report = """# Llama2-7B-Chat measured Pareto validation

Quality is the real 100-block WikiText pooled ΔNLL; latency is the median of repeated fresh-process vLLM measurements. Lower is better on both axes.

![Measured Pareto](measured_pareto.png)

## Prefill-only

Protocol: batch 8, input 2048, `gpu_memory_utilization=0.9`. The uniform references use the established baseline artifact; selected policies use the same prefill benchmark protocol. `globally_pareto_kept` is computed directly from the displayed measured numbers.

""" + markdown_table(prefill, fields) + """

## Prefill-decode

Protocol: batch 16, input 2048, output 80, phase switch enabled, `gpu_memory_utilization=0.9`. Ours uses the historical formal phase runner and 10 repeated fresh-process measurements for output lengths 1 and 80. `dense_bf16` is the corresponding historical formal baseline summary. Point 8 is excluded because it OOMs under this formal configuration.

""" + markdown_table(decode, fields) + """

## Readout

- Decode: points 0 and 3 are dominated by dense-bf16 in the formal remeasurement. Points 6 and 11 form the measured mixed-policy frontier. Point 11 is exactly the previously exported max-speed strategy (prefill dense-NVFP4; decode W4A16), verified by identical per-module phase assignments.
- Prefill-only: real ours points 4, 8, and 16 form the retained heterogeneous part of the frontier. Dense-bf16, dense-nvfp4, and sparse-nvfp4 remain valid uniform boundary anchors; these are retained transparently rather than being incorrectly claimed as dominated.
"""
    (out / "README.md").write_text(report)
    print(out / "README.md")


if __name__ == "__main__":
    main()
