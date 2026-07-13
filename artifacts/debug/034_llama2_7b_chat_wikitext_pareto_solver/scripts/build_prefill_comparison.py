#!/usr/bin/env python3
"""Combine measured prefill candidates with the existing matching baselines."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation = root / "validation" / "prefill_only"
    rows: list[dict[str, object]] = []
    for point in json.loads((validation / "selection.json").read_text()):
        index = str(point["point_index"])
        nll = read_rows(validation / f"nll_point_{index}.csv")[0]
        runs = validation / "speed" / f"point_{index}" / "runs"
        e2e = statistics.median(json.loads((runs / f"measured_{i}.json").read_text())["elapsed_ms"] for i in range(5))
        rows.append({"family": "ours", "label": f"ours_point_{index}", "policy_id": f"selected_{index}",
                     "measured_wikitext_delta_nll": float(nll["target_delta_nll"]), "e2e_median_ms": e2e})

    nll = {row["policy_id"]: row for row in read_rows(
        root.parent / "033_llama2_7b_chat_wikitext_phase_nll_proxy" / "nll" / "prefill_only.csv")}
    baseline = {row["method"]: row for row in read_rows(
        root.parents[1] / "exports" / "vllm" / "baselines" / "llama2-7b-chat" / "results" / "summary" / "speed_summary.csv")
        if row["scenario"] == "prefill_only"}
    references = {
        "dense_bf16": "p00", "dense_nvfp4": "p01", "sparse_bf16": "p02",
        "sparse_nvfp4": "p03", "marlin_nvfp4": "p04",
    }
    for method, policy_id in references.items():
        row = baseline[method]
        rows.append({"family": "uniform_reference", "label": method, "policy_id": policy_id,
                     "measured_wikitext_delta_nll": float(nll[policy_id]["target_delta_nll"]),
                     "e2e_median_ms": float(row["e2e_median_ms"])})
    for row in rows:
        q, t = row["measured_wikitext_delta_nll"], row["e2e_median_ms"]
        row["globally_pareto_kept"] = not any(
            other is not row and other["measured_wikitext_delta_nll"] <= q and other["e2e_median_ms"] <= t
            and (other["measured_wikitext_delta_nll"] < q or other["e2e_median_ms"] < t) for other in rows)
    out = validation / "measured_comparison.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
