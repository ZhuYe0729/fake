#!/usr/bin/env python3
"""Audit the one-variable overlap ablation before running GPU measurements."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
POLICY = (ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto"
          / "llama31_8b_instruct/pareto/policies/point_011.json")


def main() -> None:
    raw = POLICY.read_bytes()
    policy = json.loads(raw)
    method_map = policy["method_map"]
    pairs = Counter((row["prefill_method"], row["decode_method"])
                    for row in method_map.values())
    switched = sum(count for (prefill, decode), count in pairs.items()
                   if prefill != decode)
    if len(method_map) != 128 or switched != 128:
        raise RuntimeError(
            f"expected 128/128 switched Linear modules, got "
            f"{switched}/{len(method_map)}")
    audit = {
        "model": "Llama3.1-8B-Instruct",
        "scenario": {"batch": 8, "input_seq": 2048, "output_seq": 64},
        "policy": str(POLICY),
        "policy_sha256": hashlib.sha256(raw).hexdigest(),
        "linear_modules": len(method_map),
        "phase_switched_linear_modules": switched,
        "method_pairs": [
            {"prefill": pair[0], "decode": pair[1], "count": count}
            for pair, count in sorted(pairs.items())
        ],
        "formal_timing_trace": False,
        "controlled_variable": "PHASE_HETERO_SYNC_COPY",
        "variants": {
            "synchronous_copy": "1",
            "asynchronous_overlap": "0",
        },
    }
    out = EXP / "config_audit.json"
    out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

