#!/usr/bin/env python3
"""Build the prefill-decode table using the historical formal speed protocol."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def nll(path: Path) -> float:
    with path.open(newline="") as handle:
        return float(next(csv.DictReader(handle))["target_delta_nll"])


def median_elapsed(runs: Path, output: int) -> float:
    values = [json.loads((runs / f"measured_{i}_o{output}.json").read_text())["elapsed_ms"]
              for i in range(10)]
    return statistics.median(values)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation = root / "validation" / "prefill_decode"
    # Historical baseline summary, measured by the same phase-runtime protocol.
    dense_e2e_ms = 4868.067817762494
    rows: list[dict[str, object]] = [{
        "family": "uniform_reference", "label": "dense_bf16", "policy_id": "dense_bf16",
        "measured_wikitext_delta_nll": 0.0, "e2e_median_ms": dense_e2e_ms,
        "ttft_median_ms": "", "tpot_ms": "", "globally_pareto_kept": True,
    }]
    for point in (0, 3, 6, 11):
        runs = validation / "speed_official" / f"point_{point}" / "runs"
        e2e = median_elapsed(runs, 80)
        ttft = median_elapsed(runs, 1)
        rows.append({
            "family": "ours", "label": f"ours_point_{point}", "policy_id": f"selected_{point}",
            "measured_wikitext_delta_nll": nll(validation / f"nll_point_{point}.csv"),
            "e2e_median_ms": e2e, "ttft_median_ms": ttft,
            "tpot_ms": (e2e - ttft) / 79,
            "globally_pareto_kept": False,
        })
    for row in rows:
        loss, latency = float(row["measured_wikitext_delta_nll"]), float(row["e2e_median_ms"])
        row["globally_pareto_kept"] = not any(
            other is not row
            and float(other["measured_wikitext_delta_nll"]) <= loss
            and float(other["e2e_median_ms"]) <= latency
            and (float(other["measured_wikitext_delta_nll"]) < loss
                 or float(other["e2e_median_ms"]) < latency)
            for other in rows
        )
    out = validation / "measured_comparison_official.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
