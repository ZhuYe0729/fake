#!/usr/bin/env python3
"""Merge per-policy real-vLLM NLL JSON files into calibrated labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--reference-policy", default="p00")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-canonical-provenance", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    missing = [row["policy_id"] for row in manifest
               if not (args.raw_dir / f"{row['policy_id']}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} NLL results: {missing[:8]}")
    reference = json.loads((args.raw_dir / f"{args.reference_policy}.json").read_text())["avg_nll"]
    rows = []
    for item in manifest:
        policy_id = item["policy_id"]
        result = json.loads((args.raw_dir / f"{policy_id}.json").read_text())
        if args.require_canonical_provenance:
            provenance_path = args.raw_dir / f"{policy_id}.export_provenance.json"
            if not provenance_path.is_file():
                raise FileNotFoundError(provenance_path)
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("prune") is not False:
                raise RuntimeError(f"{policy_id}: runtime pruning or unknown provenance")
        rows.append({
            "policy_id": policy_id, "split": item["split"],
            "policy_kind": item.get("policy_kind", "unspecified"),
            "sample_count": result.get("runtime", {}).get("blocks", result.get("blocks", "")),
            "token_count": result.get("token_count", ""),
            "avg_nll": result["avg_nll"],
            "target_delta_nll": float(result["avg_nll"]) - float(reference),
            "elapsed_seconds": result.get("elapsed_seconds", ""),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "policies": len(rows),
                      "reference_policy": args.reference_policy,
                      "reference_avg_nll": reference}, indent=2))


if __name__ == "__main__":
    main()
