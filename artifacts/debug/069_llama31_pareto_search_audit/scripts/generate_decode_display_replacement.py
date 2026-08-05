#!/usr/bin/env python3
"""Generate one fixed decode neighborhood replacement for display use."""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from config import DECODE, parent_path
from generate_candidates import (
    action_tables,
    load_policy,
    map_from_pairs,
    modules_from,
    write_policy,
)


REPLACEMENT_FOR = "neighborhood_009"
MUTATION_RATE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    parent = load_policy(parent_path("prefill_decode", args.parent))
    modules = modules_from(parent)
    legal, _, _ = action_tables("prefill_decode")
    rng = random.Random(args.seed)
    pairs = {
        name: (
            parent["method_map"][name]["prefill_method"],
            parent["method_map"][name]["decode_method"],
        )
        for name in modules
    }
    selected = rng.sample(
        modules, max(1, round(len(modules) * MUTATION_RATE))
    )
    mutations = []
    for name in selected:
        old = pairs[name]
        new = rng.choice([pair for pair in legal[name] if pair != old])
        pairs[name] = new
        mutations.append({"module": name, "from": old, "to": new})

    row = write_policy(
        DECODE,
        "prefill_decode",
        "neighborhood",
        args.policy_id,
        map_from_pairs(pairs),
        {
            "seed": args.seed,
            "parent": args.parent,
            "mutation_rate": MUTATION_RATE,
            "replacement_for": REPLACEMENT_FOR,
            "display_only": True,
        },
    )
    existing_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (DECODE / "policies").glob("*/*.json")
        if path.name != f"{args.policy_id}.json"
    }
    assert row["policy_sha256"] not in existing_hashes
    manifest = {
        **row,
        "replacement_for": REPLACEMENT_FOR,
        "mutations": mutations,
    }
    path = DECODE / f"search/display_replacements/{args.policy_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
