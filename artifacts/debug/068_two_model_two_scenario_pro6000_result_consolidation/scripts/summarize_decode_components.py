#!/usr/bin/env python3
"""Summarize isolated-process TTFT/TPOT/E2E measurements for the paper table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1]
MEASURE = BUNDLE / "measurements/decode_components"


def main() -> None:
    with (BUNDLE / "data/selected_results.csv").open(newline="") as handle:
        selected = [row for row in csv.DictReader(handle) if row["scenario"] == "prefill_decode"]
    raw = {}
    for row in selected:
        key = (row["model"], row["source_label"])
        path = MEASURE / "runs" / key[0] / key[1] / "summary.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if not value.get("rtx5090_protocol_match"):
            raise RuntimeError(f"protocol mismatch: {path}")
        raw[key] = value
    selected = [row for row in selected if (row["model"], row["source_label"]) in raw]
    if not selected:
        raise RuntimeError("no complete component measurements")
    rows = []
    for row in selected:
        key = (row["model"], row["source_label"])
        value = raw[key]
        dense_key = (row["model"], "uniform_p00")
        if dense_key not in raw:
            raise RuntimeError(f"BF16 must be measured first for {row['model']}")
        dense = raw[dense_key]
        uuids = value["cuda_device_uuids"]
        if len(uuids) != 1:
            raise RuntimeError(f"multiple GPU UUIDs for {key}: {uuids}")
        rows.append({
            "model": row["model"], "table_role": row["table_role"],
            "source_label": row["source_label"], "policy_sha256": value["policy_sha256"],
            "ttft_median_ms": value["ttft_median_ms"], "tpot_ms": value["tpot_ms"],
            "e2e_median_ms": value["e2e_median_ms"],
            "ttft_speedup_vs_bf16": dense["ttft_median_ms"] / value["ttft_median_ms"],
            "tpot_speedup_vs_bf16": dense["tpot_ms"] / value["tpot_ms"],
            "e2e_speedup_vs_bf16": dense["e2e_median_ms"] / value["e2e_median_ms"],
            "cuda_device_uuid": uuids[0],
            "execution": value["execution"],
            "warmups_per_phase": value["warmup_iters_per_phase"],
            "measured_per_phase": value["measured_iters_per_phase"],
        })
    path = MEASURE / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Pro 6000 decode component timing", "",
        "RTX 5090-compatible protocol: fresh vLLM process/model per sample; O=1 and O=64 each use 1 warmup + 5 measured runs; TPOT=(median E2E-median TTFT)/63.", "",
        "| Model | Method | Policy | TTFT ms | TPOT ms | E2E ms | TTFT speedup | TPOT speedup | E2E speedup |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['table_role']} | {row['source_label']} | {float(row['ttft_median_ms']):.3f} | {float(row['tpot_ms']):.3f} | {float(row['e2e_median_ms']):.3f} | {float(row['ttft_speedup_vs_bf16']):.3f}× | {float(row['tpot_speedup_vs_bf16']):.3f}× | {float(row['e2e_speedup_vs_bf16']):.3f}× |")
    (MEASURE / "summary.md").write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
