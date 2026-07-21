#!/usr/bin/env python3
"""Turn a phase policy into the all-phase dense-NVFP4 control policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.input.read_text())
    policy["policy_id"] = "q128_phase_both_dense_nvfp4"
    policy["default_prefill_method"] = "dense_nvfp4"
    policy["default_decode_method"] = "dense_nvfp4"
    for methods in policy["method_map"].values():
        methods["prefill_method"] = "dense_nvfp4"
        methods["decode_method"] = "dense_nvfp4"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
