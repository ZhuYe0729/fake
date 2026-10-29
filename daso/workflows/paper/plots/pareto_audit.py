#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"ours": "#1f2937", "uniform": "#dc2626", "random": "#4c78a8",
          "neighborhood": "#54a24b", "suspicious": "#f28e2b", "leverage": "#f28e2b"}
MARKERS = {"ours": "o", "uniform": "s", "random": "o", "neighborhood": "^",
           "suspicious": "D", "leverage": "D"}


def frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted([row for row in rows if not any(
        other["speedup"] >= row["speedup"] and other["quality"] >= row["quality"]
        and (other["speedup"] > row["speedup"] or other["quality"] > row["quality"])
        for other in rows if other is not row)], key=lambda row: row["speedup"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("prefill_only", "prefill_decode"), required=True)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[3] / "results/paper")
    args = parser.parse_args()
    report = args.root / "appendix_pareto_audit" / args.scenario / "report"
    with (report / "all_results.csv").open(newline="") as handle:
        rows = [{**row, "speedup": float(row["speedup"]), "quality": float(row["quality"])}
                for row in csv.DictReader(handle)]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for family in sorted({row["family"] for row in rows}):
        subset = [row for row in rows if row["family"] == family]
        ax.scatter([row["speedup"] for row in subset], [row["quality"] for row in subset],
                   marker=MARKERS.get(family, "o"), color=COLORS.get(family, "#777777"),
                   label=family.replace("suspicious", "leverage"), alpha=.85)
    edge = frontier(rows)
    ax.plot([row["speedup"] for row in edge], [row["quality"] for row in edge],
            "--", color="#111827", linewidth=1.3, label="combined frontier")
    ax.set_xlabel("Measured E2E speedup vs. BF16")
    ax.set_ylabel("ARC-C acc_norm" if args.scenario == "prefill_only" else "IWSLT SacreBLEU")
    ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    output = args.root / "figures" / f"pareto_search_audit_{args.scenario}"
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf")); plt.close(fig)


if __name__ == "__main__":
    main()

