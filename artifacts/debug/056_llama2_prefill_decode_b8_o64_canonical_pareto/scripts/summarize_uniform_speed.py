#!/usr/bin/env python3
"""Summarize formal same-runtime uniform B=8/O=64 speed baselines."""
from __future__ import annotations

import csv
from pathlib import Path

from scenario import EXP


BASELINES = (
    ("dense_bf16", "p00", "dense BF16 in both phases"),
    ("dense_nvfp4", "p01", "dense NVFP4 in both phases"),
    ("sparse_bf16", "p02", "canonical sparse BF16 in both phases"),
    ("sparse_nvfp4_legal_projection", "p03",
     "prefill sparse NVFP4; decode dense NVFP4 because sparse NVFP4 is unsupported at M=8"),
    ("w4a16_marlin", "p04", "W4A16/Marlin NVFP4 in both phases"),
)


def summary(policy_id: str) -> dict[str, str]:
    path = EXP / "speed/runs" / policy_id / "summary.csv"
    rows = list(csv.DictReader(path.open()))
    if len(rows) != 1 or rows[0]["status"] != "OK":
        raise RuntimeError(f"invalid formal speed result: {policy_id}")
    return rows[0]


def main() -> None:
    dense = summary("p00")
    dense_ms = float(dense["e2e_median_ms"])
    rows = []
    for method, policy_id, note in BASELINES:
        result = summary(policy_id)
        median = float(result["e2e_median_ms"])
        rows.append({"method": method, "policy_id": policy_id, "e2e_median_ms": median,
                     "speedup_vs_dense_bf16": dense_ms / median,
                     "ttft_median_ms": float(result["ttft_median_ms"]),
                     "tpot_ms": float(result["tpot_ms"]), "note": note,
                     "runtime": "phase_hetero_mytest; VLLM V1; BF16 KV; chunked prefill disabled"})
    output = EXP / "speed/uniform_baselines.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Uniform speed baselines (same runtime)", "",
             "All rows use B=8, input=2048, output=64, `phase_hetero_mytest`, VLLM V1, BF16 KV, and disabled chunked prefill.", "",
             "| method | E2E median ms | speedup | TTFT ms | TPOT ms | note |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['e2e_median_ms']:.3f} | {row['speedup_vs_dense_bf16']:.3f} | {row['ttft_median_ms']:.3f} | {row['tpot_ms']:.3f} | {row['note']} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
