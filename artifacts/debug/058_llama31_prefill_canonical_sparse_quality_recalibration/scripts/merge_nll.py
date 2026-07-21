#!/usr/bin/env python3
"""Merge canonical phase-vLLM prefill NLL labels into a deterministic CSV."""
from __future__ import annotations
import csv
import json
from scenario import EXP

def main() -> None:
    manifest = json.loads((EXP / "policies/prefill_only/manifest.json").read_text())
    raw = {}
    for item in manifest:
        path = EXP / "nll/raw" / f"{item['policy_id']}.json"
        if not path.exists(): raise RuntimeError(f"missing NLL: {item['policy_id']}")
        raw[item["policy_id"]] = json.loads(path.read_text())
    base = float(raw["p00"]["avg_nll"])
    rows = [{"policy_id": item["policy_id"], "split": item["split"], "policy_kind": item["policy_kind"],
             "sample_count": raw[item["policy_id"]]["runtime"].get("blocks", 100) if isinstance(raw[item["policy_id"]].get("runtime"), dict) else 100,
             "token_count": raw[item["policy_id"]]["token_count"], "avg_nll": raw[item["policy_id"]]["avg_nll"],
             "target_delta_nll": float(raw[item["policy_id"]]["avg_nll"]) - base,
             "elapsed_seconds": raw[item["policy_id"]]["elapsed_seconds"]} for item in manifest]
    output = EXP / "nll/prefill_only.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)
if __name__ == "__main__": main()
