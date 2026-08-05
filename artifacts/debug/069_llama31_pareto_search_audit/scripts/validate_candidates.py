#!/usr/bin/env python3
"""Statically audit generated search policies and their action legality."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter

from config import DECODE, EXP_057, KERNEL, METHODS, PREFILL, SOURCE_038, TYPES


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def legal_actions(scenario: str):
    source = SOURCE_038 / "action_support.csv" if scenario == "prefill_only" else EXP_057 / "speed/action_support.csv"
    data = rows(source)
    if scenario == "prefill_only":
        supported = {(row["module_name"], row["kernel"]) for row in data if row["supported"] == "True"}
        return lambda name, pre, dec: pre == dec and (name, KERNEL[pre]) in supported
    supported = {(row["phase"], row["module_name"], row["kernel"]) for row in data if row["supported"] == "True"}
    return lambda name, pre, dec: (
        ("prefill", name, KERNEL[pre]) in supported
        and ("decode", name, KERNEL[dec]) in supported
    )


def validate(scenario: str, root):
    candidates = rows(root / "search/candidates.csv")
    counts = Counter(row["family"] for row in candidates)
    assert counts == {"random": 15, "neighborhood": 10, "suspicious": 5}, counts
    assert len({row["policy_id"] for row in candidates}) == 30
    hashes = set()
    legal = legal_actions(scenario)
    for row in candidates:
        path = root.parent.parent.parent.parent / row["policy_json"] if not row["policy_json"].startswith("/") else None
        if path is None:
            from pathlib import Path
            path = Path(row["policy_json"])
        payload = json.loads(path.read_text())
        assert payload["scenario"] == scenario
        assert payload["modules_to_not_convert"] == ["lm_head"]
        mapping = payload["method_map"]
        assert len(mapping) == 128
        for name, pair in mapping.items():
            assert name.rsplit(".", 1)[-1] in TYPES
            assert pair["prefill_method"] in METHODS
            assert pair["decode_method"] in METHODS
            assert legal(name, pair["prefill_method"], pair["decode_method"]), (name, pair)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["policy_sha256"]
        assert digest not in hashes, f"duplicate policy {row['policy_id']}"
        hashes.add(digest)
    return {"scenario": scenario, "candidates": len(candidates), "families": dict(counts)}


def main():
    print(json.dumps({
        "prefill_only": validate("prefill_only", PREFILL),
        "prefill_decode": validate("prefill_decode", DECODE),
    }, indent=2))


if __name__ == "__main__":
    main()
