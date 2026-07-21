#!/usr/bin/env python3
"""Validate 054 results and build the fixed-policy phase NLL table."""
from __future__ import annotations

import csv
import os
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((EXPERIMENT / "policies/prefill_only/manifest.json").read_text())
    sample_hash = sha256(EXPERIMENT / "samples/wikitext_2048_targets.pt")
    results = {}
    for row in manifest:
        result = json.loads((EXPERIMENT / "results" / f"{row['policy_id']}.json").read_text())
        runtime = result["runtime"]
        if (len(result["blocks"]) != 100 or result["token_count"] != 204800
                or runtime.get("sample_sha256") != sample_hash
                or runtime.get("policy_sha256") != row["sha256"]):
            raise RuntimeError(f"invalid result: {row['policy_id']}")
        if row["policy_id"] != "p00" and (not runtime.get("phase_hetero")
                                             or runtime.get("checkpoint_config", {}).get("quant_method") != "phase_hetero_mytest"):
            raise RuntimeError(f"non-phase compressed result: {row['policy_id']}")
        results[row["policy_id"]] = result
    reference = results["p00"]["avg_nll"]
    rows = [{"policy_id": row["policy_id"], "split": row["split"], "policy_kind": row["policy_kind"],
             "sample_count": len(results[row["policy_id"]]["blocks"]),
             "token_count": results[row["policy_id"]]["token_count"],
             "avg_nll": results[row["policy_id"]]["avg_nll"],
             "target_delta_nll": results[row["policy_id"]]["avg_nll"] - reference,
             "elapsed_seconds": results[row["policy_id"]]["elapsed_seconds"]}
            for row in manifest]
    output = EXPERIMENT / "nll/prefill_only.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
