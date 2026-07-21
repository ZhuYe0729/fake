#!/usr/bin/env python3
"""Validate per-policy results and write the real-vLLM NLL label table."""
from __future__ import annotations

import argparse
import csv
import json
import math

from common import MODELS, model_root, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    root = model_root(args.model)
    manifest = json.loads((root / "policies/prefill_only/manifest.json").read_text())
    sample_hash = sha256(root / "samples/wikitext_2048_targets.pt")
    payloads = {}
    for row in manifest:
        path = root / "results" / f"{row['policy_id']}.json"
        payload = json.loads(path.read_text())
        if payload["runtime"].get("sample_sha256") != sample_hash:
            raise RuntimeError(f"sample mismatch: {row['policy_id']}")
        if len(payload["blocks"]) != 100 or payload["token_count"] != 100 * 2048:
            raise RuntimeError(f"incomplete result: {row['policy_id']}")
        if row["policy_id"] >= "p05" and payload["runtime"].get("policy_sha256") != row["sha256"]:
            raise RuntimeError(f"policy mismatch: {row['policy_id']}")
        payloads[row["policy_id"]] = payload
    dense = payloads["p00"]
    labels = []
    for row in manifest:
        result = payloads[row["policy_id"]]
        labels.append({"policy_id": row["policy_id"], "split": row["split"], "policy_kind": row["policy_kind"], "sample_count": len(result["blocks"]), "token_count": result["token_count"], "avg_nll": result["avg_nll"], "perplexity": result["perplexity"], "target_delta_nll": result["avg_nll"] - dense["avg_nll"], "target_delta_log_ppl": math.log(result["perplexity"]) - math.log(dense["perplexity"]), "elapsed_seconds": result["elapsed_seconds"]})
    out = root / "nll/prefill_only.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader(); writer.writerows(labels)
    print(out)


if __name__ == "__main__":
    main()
