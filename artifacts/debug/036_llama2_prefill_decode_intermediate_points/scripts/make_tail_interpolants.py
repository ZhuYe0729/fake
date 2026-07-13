#!/usr/bin/env python3
"""Split the final 11-module 34->35 policy jump into three policies."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "prefill_decode/pareto/policies"
STEPS = {36: 3, 37: 6, 38: 9}


def main() -> None:
    low = json.loads((POLICIES / "point_034.json").read_text())
    high = json.loads((POLICIES / "point_035.json").read_text())
    changed = [name for name in low["method_map"] if low["method_map"][name] != high["method_map"][name]]
    if len(changed) != 11:
        raise RuntimeError(f"expected 11 final-jump modules, got {len(changed)}")
    for point, count in STEPS.items():
        policy = json.loads(json.dumps(low))
        for name in changed[-count:]:
            policy["method_map"][name] = high["method_map"][name]
        (POLICIES / f"point_{point:03d}.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        print(f"point_{point:03d}: switched {count}/11 final-jump modules")


if __name__ == "__main__":
    main()
