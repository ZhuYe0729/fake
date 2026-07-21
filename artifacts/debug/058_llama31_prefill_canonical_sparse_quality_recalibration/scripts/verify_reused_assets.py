#!/usr/bin/env python3
"""Verify hashes and coverage of the immutable assets imported from 057."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from scenario import EXP, CANONICAL, LOCAL_ERRORS, METHODS, TYPES

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def main() -> None:
    manifest = json.loads((EXP / "provenance.json").read_text())["canonical_manifest"]["states"]
    states = {}
    for method in ("sparse_bf16", "sparse_nvfp4"):
        path = CANONICAL / method / "model.pt"
        actual = digest(path)
        if actual != manifest[method]["sha256"]:
            raise RuntimeError(f"canonical hash mismatch for {method}")
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
        linear = [name for name in state if name.endswith(".weight") and ".layers." in name]
        states[method] = {"sha256": actual, "linear_weights": len(linear)}
    errors = {}
    for method in METHODS[1:]:
        rows = (LOCAL_ERRORS / f"prefill_{method}.csv").read_text().splitlines()
        if len(rows) != 17:
            raise RuntimeError(f"incomplete prefill local error table for {method}: {len(rows)-1} rows")
        errors[method] = {"rows": len(rows)-1, "expected_cells": 4 * len(TYPES)}
    result = {"canonical": states, "prefill_local_errors": errors,
              "result": "verified reusable canonical states and prefill-only local errors"}
    output = EXP / "canonical/verification.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
