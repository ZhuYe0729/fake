#!/usr/bin/env python3
"""Paper-style speedup-versus-quality Pareto plots from measured results."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


NAVY = "#202B3C"
RED = "#D62728"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def draw(
    *, title: str, rows: list[dict[str, str]], dense_latency_ms: float,
    ours_labels: list[str], baseline_labels: list[str], output: Path,
    offscale_labels: list[str] | None = None,
    max_speed_label: str,
) -> None:
    by_label = {row["label"]: row for row in rows}
    ours = [by_label[label] for label in ours_labels]
    baselines = [by_label[label] for label in baseline_labels]
    offscale = [by_label[label] for label in (offscale_labels or [])]

    def x(row: dict[str, str]) -> float:
        return dense_latency_ms / f(row, "e2e_median_ms")

    def y(row: dict[str, str]) -> float:
        return -f(row, "measured_wikitext_delta_nll")

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 20, "axes.labelsize": 16})
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    ax.plot([x(row) for row in ours], [y(row) for row in ours], "-o", color=NAVY,
            linewidth=3, markersize=11, label="Mixed Pareto policies", zorder=3)
    endpoint = by_label[max_speed_label]
    ax.scatter(x(endpoint), y(endpoint), marker="*", s=300, color="#F0A202",
               edgecolor=NAVY, linewidth=1.2, zorder=5, label="Ours max-speed")
    ax.annotate("Ours max-speed", (x(endpoint), y(endpoint)), xytext=(10, 10),
                textcoords="offset points", color="#8A5800", fontsize=13, fontweight="bold")
    ax.scatter([x(row) for row in baselines], [y(row) for row in baselines], marker="s",
               s=185, color=RED, label="Uniform methods", zorder=4)
    display = {
        "dense_bf16": "dense BF16", "dense_nvfp4": "dense NVFP4",
        "marlin_nvfp4": "Marlin NVFP4", "sparse_bf16": "sparse BF16",
        "sparse_nvfp4": "sparse NVFP4", "w4a16_ours": "W4A16",
        "sparse_nvfp4_prefill_dense_nvfp4_decode": "sparse NVFP4 → dense NVFP4",
    }
    for row in baselines:
        label = display.get(row["label"], row["label"])
        offset = (10, 10) if y(row) < -2.5 else (10, -18)
        ax.annotate(label, (x(row), y(row)), xytext=offset, textcoords="offset points",
                    color="#B51F24", fontsize=13)

    ax.set_title(title)
    ax.set_xlabel("Measured E2E speedup vs dense BF16 (higher is better)")
    ax.set_ylabel("WikiText quality proxy: −ΔNLL (higher is better)")
    ax.grid(alpha=0.28)
    ax.legend(loc="upper right" if offscale else "lower left", frameon=True)
    ax.margins(x=0.07, y=0.12)
    if offscale:
        bottom, top = ax.get_ylim()
        for row in offscale:
            ax.scatter(x(row), bottom + 0.05 * (top - bottom), marker="v", s=175, color=RED, zorder=4)
            ax.annotate(f"{display.get(row['label'], row['label'])}\n−ΔNLL={-y(row):.1f} (off-scale)",
                        (x(row), bottom + 0.05 * (top - bottom)), xytext=(12, -32),
                        textcoords="offset points", color="#B51F24", fontsize=11)
    fig.savefig(output, dpi=260)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "report"
    prefill = read_rows(root / "validation" / "prefill_only" / "measured_comparison.csv")
    decode = read_rows(root / "validation" / "prefill_decode" / "measured_comparison_official.csv")
    prefill_dense = next(row for row in prefill if row["label"] == "dense_bf16")
    decode_dense = next(row for row in decode if row["label"] == "dense_bf16")

    draw(
        title="Llama2-7B prefill-only: speedup vs WikiText quality",
        rows=prefill, dense_latency_ms=f(prefill_dense, "e2e_median_ms"),
        ours_labels=["ours_point_4", "ours_point_8", "ours_point_16"],
        baseline_labels=["dense_bf16", "marlin_nvfp4", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4"],
        max_speed_label="ours_point_16",
        output=out / "pareto_speedup_vs_wikitext_prefill_only.png",
    )
    draw(
        title="Llama2-7B prefill-decode: speedup vs WikiText quality",
        rows=decode, dense_latency_ms=f(decode_dense, "e2e_median_ms"),
        ours_labels=["ours_point_6", "ours_point_11"],
        baseline_labels=["dense_bf16"],
        max_speed_label="ours_point_11",
        output=out / "pareto_speedup_vs_wikitext_prefill_decode.png",
    )
    print(out / "pareto_speedup_vs_wikitext_prefill_only.png")
    print(out / "pareto_speedup_vs_wikitext_prefill_decode.png")


if __name__ == "__main__":
    main()
