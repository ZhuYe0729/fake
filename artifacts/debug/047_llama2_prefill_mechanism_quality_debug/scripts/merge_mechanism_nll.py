#!/usr/bin/env python3
"""Validate and merge mechanism-policy NLL results."""
from __future__ import annotations

import csv
import json

from common import DEBUG, SOURCE, sha256


def main() -> None:
    manifest = json.loads((DEBUG / "manifest.json").read_text())
    dense = json.loads((SOURCE / "results/p00.json").read_text())["avg_nll"]
    sample_hash = sha256(SOURCE / "samples/wikitext_2048_targets.pt")
    rows = []
    for item in manifest:
        payload = json.loads((DEBUG / "results" / f"{item['policy_id']}.json").read_text())
        runtime = payload["runtime"]
        if len(payload["blocks"]) != 100 or payload["token_count"] != 204800:
            raise RuntimeError(f"incomplete NLL: {item['policy_id']}")
        if runtime["sample_sha256"] != sample_hash or runtime["policy_sha256"] != item["sha256"]:
            raise RuntimeError(f"provenance mismatch: {item['policy_id']}")
        rows.append({**item, "avg_nll": payload["avg_nll"], "delta_nll": payload["avg_nll"] - dense, "perplexity": payload["perplexity"], "elapsed_seconds": payload["elapsed_seconds"]})
    out = DEBUG / "nll.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
