#!/usr/bin/env python3
"""Merge validated intermediate points into a clearly marked debug NLL frontier."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT.parent / "035_llama2_prefill_decode_e2e_speed_model"
BASELINE_MS = 5150.02822317183
INTERMEDIATE = (34, 36, 37, 38)


def main() -> None:
    rows = list(csv.DictReader((BASE_ROOT / "report/formal_util085_actual_nll_summary.csv").open()))
    for point in INTERMEDIATE:
        runs = ROOT / "formal_util085" / f"point_{point}" / "runs"
        values = [json.loads(path.read_text())["elapsed_ms"] for path in runs.glob("measured_*_o80.json")]
        kept = [value for value in values if value < 10_000]
        nll = next(csv.DictReader((ROOT / "actual_nll" / f"point_{point}.csv").open()))
        rows.append({"point": f"i{point}", "source_point": point, "measured_wikitext_delta_nll": nll["target_delta_nll"],
                     "e2e_median_ms": statistics.median(kept), "e2e_min_ms": min(kept), "e2e_max_ms": max(kept),
                     "samples": len(kept), "unstable": "screened_stall", "speedup_vs_point0": BASELINE_MS / statistics.median(kept)})
    for row in rows:
        row.setdefault("source_point", row["point"])
        row["speedup_vs_point0"] = float(row["speedup_vs_point0"])
        row["measured_wikitext_delta_nll"] = float(row["measured_wikitext_delta_nll"])
        row["e2e_median_ms"] = float(row["e2e_median_ms"])
    stable = [row for row in rows if str(row.get("unstable", "False")) == "False" or row.get("unstable") == "screened_stall"]
    for row in rows:
        row["pareto_kept"] = row in stable and not any(
            other is not row and other in stable
            and other["e2e_median_ms"] <= row["e2e_median_ms"]
            and other["measured_wikitext_delta_nll"] <= row["measured_wikitext_delta_nll"]
            and (other["e2e_median_ms"] < row["e2e_median_ms"] or other["measured_wikitext_delta_nll"] < row["measured_wikitext_delta_nll"])
            for other in rows)
    out = ROOT / "report"; out.mkdir(exist_ok=True)
    fields = ["point", "source_point", "measured_wikitext_delta_nll", "e2e_median_ms", "e2e_min_ms", "e2e_max_ms", "samples", "unstable", "speedup_vs_point0", "pareto_kept"]
    with (out / "intermediate_actual_nll_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    frontier = sorted((row for row in rows if row["pareto_kept"]), key=lambda row: row["speedup_vs_point0"])
    other = [row for row in rows if not row["pareto_kept"] and row.get("unstable") != "True"]
    old_unstable = [row for row in rows if row.get("unstable") == "True"]
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.plot([row["speedup_vs_point0"] for row in frontier], [-row["measured_wikitext_delta_nll"] for row in frontier], "-o", color="#202B3C", linewidth=3, markersize=8, label="Measured/screened frontier")
    ax.scatter([row["speedup_vs_point0"] for row in other], [-row["measured_wikitext_delta_nll"] for row in other], marker="x", s=85, color="#8292A8", label="Dominated")
    ax.scatter([row["speedup_vs_point0"] for row in old_unstable], [-row["measured_wikitext_delta_nll"] for row in old_unstable], marker="X", s=130, color="#8E44AD", label="Old point 9 (unstable)")
    added = [row for row in rows if str(row["point"]).startswith("i")]
    ax.scatter([row["speedup_vs_point0"] for row in added], [-row["measured_wikitext_delta_nll"] for row in added], marker="D", s=80, color="#0F766E", label="New intermediate (stall-screened)", zorder=5)
    for row in added:
        ax.annotate(f"{row['point']}\n{row['speedup_vs_point0']:.2f}x", (row["speedup_vs_point0"], -row["measured_wikitext_delta_nll"]), xytext=(6, 7), textcoords="offset points", fontsize=9, color="#0F766E")
    ax.set_title("Llama2-7B prefill-decode: measured WikiText Pareto with intermediate policies")
    ax.set_xlabel("E2E speedup vs dense BF16 / point 0 (higher is better)")
    ax.set_ylabel("Measured WikiText quality: −ΔNLL (higher is better)")
    ax.grid(alpha=.28); ax.margins(x=.07, y=.13); ax.legend(loc="lower left")
    fig.savefig(out / "pareto_speedup_vs_wikitext_with_intermediates.png", dpi=260)


if __name__ == "__main__":
    main()
