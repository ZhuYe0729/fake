#!/usr/bin/env python3
"""Validate and merge the frozen 72-policy NLL sweep."""
from __future__ import annotations

import csv
import json

from common import PROTOCOL, RUN, sha256


def main() -> None:
    manifest = json.loads((RUN / "policies/prefill_decode/manifest.json").read_text())
    if len(manifest) != 72:
        raise RuntimeError("manifest must contain 72 policies")
    sample_hash = sha256(RUN / "samples/wikitext_2048_64.pt")
    waves = (100 + PROTOCOL["batch"] - 1) // PROTOCOL["batch"]
    results = {}
    for row in manifest:
        result = json.loads((RUN / "calibration/raw" / f"{row['policy_id']}.json").read_text())
        runtime = result["runtime"]
        if len(result["blocks"]) != 100 or result["token_count"] != 100 * PROTOCOL["output_tokens"]:
            raise RuntimeError(f"invalid coverage: {row['policy_id']}")
        if runtime.get("sample_sha256") != sample_hash or runtime.get("policy_sha256") != row["sha256"]:
            raise RuntimeError(f"provenance mismatch: {row['policy_id']}")
        if (not runtime.get("phase_hetero")
                or runtime.get("quantization_config", {}).get("quant_method") != "phase_hetero_mytest"
                or runtime.get("chunked_prefill_enabled") is not False
                or runtime.get("max_num_batched_tokens") != PROTOCOL["teacher_forcing_capacity"]
                or runtime.get("phase_trace_events", {}).get("apply_prefill") != waves * 128
                or runtime.get("phase_trace_events", {}).get("apply_decode") != waves * 128 * PROTOCOL["decode_steps"]):
            raise RuntimeError(f"runtime mismatch: {row['policy_id']}")
        audit = json.loads((RUN / "calibration/audits" / f"{row['policy_id']}.json").read_text())
        if audit.get("prune") is not False or audit.get("policy_sha256") != row["sha256"]:
            raise RuntimeError(f"checkpoint audit mismatch: {row['policy_id']}")
        results[row["policy_id"]] = result
    reference = results["p00"]["avg_nll"]
    rows = [{"policy_id": row["policy_id"], "split": row["split"], "policy_kind": row["policy_kind"],
             "sample_count": 100, "token_count": results[row["policy_id"]]["token_count"],
             "avg_nll": results[row["policy_id"]]["avg_nll"],
             "target_delta_nll": results[row["policy_id"]]["avg_nll"] - reference,
             "elapsed_seconds": results[row["policy_id"]]["elapsed_seconds"]} for row in manifest]
    output = RUN / "calibration/nll/prefill_decode.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
