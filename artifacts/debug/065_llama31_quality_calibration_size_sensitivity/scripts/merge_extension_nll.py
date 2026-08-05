#!/usr/bin/env python3
"""Validate and merge extension NLL labels against the frozen p00 reference."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"


def main() -> None:
    manifest = json.loads((OUT / "extension/policies/manifest.json").read_text())
    raw = OUT / "extension/nll/raw"
    missing = [row["policy_id"] for row in manifest if not (raw / f"{row['policy_id']}.json").exists()]
    if missing:
        raise RuntimeError(f"missing extension NLL labels: {missing}")
    reference = json.loads((SOURCE / "nll/raw/p00.json").read_text())["avg_nll"]
    rows = []
    for item in manifest:
        policy_id = item["policy_id"]
        payload = json.loads((raw / f"{policy_id}.json").read_text())
        provenance = json.loads((raw / f"{policy_id}.export_provenance.json").read_text())
        trace = payload["runtime"].get("phase_trace_events", {})
        if provenance.get("prune") is not False:
            raise RuntimeError(f"non-canonical export: {policy_id}")
        if not trace.get("enter_decode") or not trace.get("apply_decode"):
            raise RuntimeError(f"missing phase switch: {policy_id}")
        runtime = payload["runtime"]
        if runtime.get("input_tokens") != 2048 or runtime.get("decode_tokens") != 64 or runtime.get("batch_size") != 8:
            raise RuntimeError(f"wrong NLL protocol: {policy_id}")
        rows.append({
            "policy_id": policy_id,
            "policy_kind": item["policy_kind"],
            "source_index": item["source_index"],
            "sample_count": runtime["blocks"],
            "token_count": payload["token_count"],
            "avg_nll": payload["avg_nll"],
            "target_delta_nll": payload["avg_nll"] - reference,
            "elapsed_seconds": payload["elapsed_seconds"],
            "decode_transitions": trace["enter_decode"],
        })
    output = OUT / "extension/nll/extension.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
