#!/usr/bin/env python3
"""Hash-check and merge the additional coverage NLL labels."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/048_llama2_prefill_quality_coverage"
SOURCE = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((DEBUG / "manifest.json").read_text())
    sample = SOURCE / "samples/wikitext_2048_targets.pt"; baseline = json.loads((SOURCE / "results/p00.json").read_text())["avg_nll"]
    rows = []
    for item in manifest:
        result = json.loads((DEBUG / "results" / f"{item['policy_id']}.json").read_text()); runtime = result["runtime"]
        if len(result["blocks"]) != 100 or result["token_count"] != 204800 or runtime["sample_sha256"] != sha256(sample) or runtime["policy_sha256"] != item["sha256"]:
            raise RuntimeError(f"invalid provenance: {item['policy_id']}")
        rows.append({**item, "avg_nll": result["avg_nll"], "delta_nll": result["avg_nll"] - baseline, "perplexity": result["perplexity"], "elapsed_seconds": result["elapsed_seconds"]})
    with (DEBUG / "nll.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(DEBUG / "nll.csv")


if __name__ == "__main__":
    main()
