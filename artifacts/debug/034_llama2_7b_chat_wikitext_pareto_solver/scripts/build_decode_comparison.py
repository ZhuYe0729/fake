#!/usr/bin/env python3
"""Combine measured selected policies and same-protocol uniform references."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def speed_row(runs: Path) -> tuple[float, float, float]:
    one = [runs / f"measured_{i}_o1.json" for i in range(10)]
    full = [runs / f"measured_{i}_o80.json" for i in range(10)]
    if not all(path.exists() for path in one + full):
        raise RuntimeError(f"incomplete speed runs: {runs}")
    ttft = 1000 * statistics.median(json.loads(path.read_text())["generate_s"] for path in one)
    e2e = 1000 * statistics.median(json.loads(path.read_text())["generate_s"] for path in full)
    return e2e, ttft, (e2e - ttft) / 79


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation = root / "validation" / "prefill_decode"
    rows: list[dict[str, object]] = []
    for point in json.loads((validation / "selection.json").read_text()):
        index = str(point["point_index"])
        nll = csv_rows(validation / f"nll_point_{index}.csv")[0]
        e2e, ttft, tpot = speed_row(validation / "speed_mem08" / f"point_{index}" / "runs")
        rows.append({
            "family": "ours", "label": f"ours_point_{index}", "policy_id": f"selected_{index}",
            "measured_wikitext_delta_nll": float(nll["target_delta_nll"]),
            "e2e_median_ms": e2e, "ttft_median_ms": ttft, "tpot_ms": tpot,
        })

    nll_by_policy = {row["policy_id"]: row for row in csv_rows(
        root.parent / "033_llama2_7b_chat_wikitext_phase_nll_proxy" / "nll" / "prefill_decode.csv")}
    references = {
        "sparse_bf16": "p02",
        "sparse_nvfp4_prefill_dense_nvfp4_decode": "p03",
        "w4a16_ours": "p04",
    }
    for label, policy_id in references.items():
        e2e, ttft, tpot = speed_row(validation / "uniform_references" / label / "runs")
        rows.append({
            "family": "uniform_reference", "label": label, "policy_id": policy_id,
            "measured_wikitext_delta_nll": float(nll_by_policy[policy_id]["target_delta_nll"]),
            "e2e_median_ms": e2e, "ttft_median_ms": ttft, "tpot_ms": tpot,
        })

    for row in rows:
        nll, e2e = row["measured_wikitext_delta_nll"], row["e2e_median_ms"]
        row["globally_pareto_kept"] = not any(
            other is not row
            and other["measured_wikitext_delta_nll"] <= nll
            and other["e2e_median_ms"] <= e2e
            and (other["measured_wikitext_delta_nll"] < nll or other["e2e_median_ms"] < e2e)
            for other in rows
        )
    out = validation / "measured_comparison_mem08.csv"
    fields = list(rows[0])
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
