#!/usr/bin/env python3
"""Render the `.85` formal decode curve with real WikiText NLL."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    actual_nll = {}
    for point in range(12):
        with (root / "actual_nll" / f"point_{point}.csv").open(newline="") as handle:
            actual_nll[point] = float(next(csv.DictReader(handle))["target_delta_nll"])
    rows = []
    for point in range(12):
        files = sorted((root / "formal_util085" / f"point_{point}" / "runs").glob("measured_*_o80.json"))
        values = [json.loads(path.read_text())["elapsed_ms"] for path in files]
        if len(values) != 10:
            continue
        rows.append({"point": point, "measured_wikitext_delta_nll": actual_nll[point], "e2e_median_ms": statistics.median(values),
                     "e2e_min_ms": min(values), "e2e_max_ms": max(values), "samples": len(values),
                     "unstable": point == 9})
    baseline = next(row["e2e_median_ms"] for row in rows if row["point"] == 0)
    for row in rows:
        row["speedup_vs_point0"] = baseline / row["e2e_median_ms"]
        row["pareto_kept"] = (not row["unstable"]) and not any(
            other is not row and not other["unstable"]
            and other["measured_wikitext_delta_nll"] <= row["measured_wikitext_delta_nll"] and other["e2e_median_ms"] <= row["e2e_median_ms"]
            and (other["measured_wikitext_delta_nll"] < row["measured_wikitext_delta_nll"] or other["e2e_median_ms"] < row["e2e_median_ms"])
            for other in rows)
    out = root / "report"; out.mkdir(exist_ok=True)
    with (out / "formal_util085_actual_nll_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    kept = [row for row in rows if row["pareto_kept"]]
    dominated = [row for row in rows if not row["pareto_kept"] and not row["unstable"]]
    unstable = next(row for row in rows if row["unstable"])
    x = lambda row: row["speedup_vs_point0"]
    y = lambda row: -row["measured_wikitext_delta_nll"]
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 19, "axes.labelsize": 16})
    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    ax.plot([x(row) for row in kept], [y(row) for row in kept], "-o", color="#202B3C", linewidth=3,
            markersize=9, label="Measured mixed-policy frontier", zorder=3)
    if dominated:
        ax.scatter([x(row) for row in dominated], [y(row) for row in dominated], marker="x", s=90,
                   linewidth=2.2, color="#8292A8", label="Measured dominated policies", zorder=2)
    ax.scatter(x(unstable), y(unstable), marker="X", s=175, color="#8E44AD", label="point 9 (unstable run)", zorder=4)
    ax.annotate("point 9: concurrent-load outlier\n(not used for frontier)", (x(unstable), y(unstable)),
                xytext=(10, 12), textcoords="offset points", color="#6C3483", fontsize=11)
    endpoint = next(row for row in rows if row["point"] == 11)
    ax.scatter(x(endpoint), y(endpoint), marker="*", s=320, color="#F0A202", edgecolor="#202B3C",
               linewidth=1.2, zorder=5, label="Ours max-speed")
    ax.annotate("Ours max-speed", (x(endpoint), y(endpoint)), xytext=(9, 10), textcoords="offset points",
                color="#8A5800", fontsize=12, fontweight="bold")
    dense = next(row for row in rows if row["point"] == 0)
    ax.annotate("dense BF16", (x(dense), y(dense)), xytext=(10, 10), textcoords="offset points", fontsize=12)
    ax.set_title("Llama2-7B prefill-decode: measured Pareto (.85 KV cache)")
    ax.set_xlabel("Measured E2E speedup vs dense BF16 / point 0 (higher is better)")
    ax.set_ylabel("Measured WikiText quality: −ΔNLL (higher is better)")
    ax.grid(alpha=.28); ax.margins(x=.07, y=.13); ax.legend(loc="lower left", frameon=True)
    fig.savefig(out / "pareto_speedup_vs_wikitext_prefill_decode_util085_actual_nll.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
