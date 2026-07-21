#!/usr/bin/env python3
"""Merge completed canonical prefill-decode NLL JSON files against p00."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat"


def main() -> None:
    manifest = json.loads((EXP / "policies/prefill_decode/manifest.json").read_text())
    raw = EXP / "nll/raw"
    missing = [row["policy_id"] for row in manifest if not (raw / f"{row['policy_id']}.json").exists()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} NLL labels: {missing[:8]}")
    reference = json.loads((raw / "p00.json").read_text())["avg_nll"]
    rows = []
    for item in manifest:
        payload = json.loads((raw / f"{item['policy_id']}.json").read_text())
        trace = payload["runtime"].get("phase_trace_events", {})
        if not trace.get("enter_decode") or not trace.get("apply_decode"):
            raise RuntimeError(f"missing phase switch: {item['policy_id']}")
        rows.append({"policy_id": item["policy_id"], "split": item["split"], "policy_kind": item["policy_kind"],
                     "sample_count": payload["runtime"]["blocks"], "token_count": payload["token_count"],
                     "avg_nll": payload["avg_nll"], "target_delta_nll": payload["avg_nll"] - reference,
                     "elapsed_seconds": payload["elapsed_seconds"], "decode_transitions": trace["enter_decode"]})
    output = EXP / "nll/prefill_decode.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
