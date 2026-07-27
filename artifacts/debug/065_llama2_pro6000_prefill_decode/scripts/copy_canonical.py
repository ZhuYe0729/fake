#!/usr/bin/env python3
"""Copy the audited 064 canonical states into 065 without recomputation."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import CANONICAL_SOURCE, RUN, sha256, write_json

METHODS = ("sparse_bf16", "sparse_nvfp4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=CANONICAL_SOURCE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = {}
    for method in METHODS:
        source_dir = args.source / method
        source = source_dir / "model.pt"
        metadata = source_dir / "metadata.json"
        if not source.is_file() or not metadata.is_file():
            raise FileNotFoundError(source_dir)
        target_dir = RUN / "canonical/prepared" / method
        target = target_dir / "model.pt"
        target_dir.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(source)
        if target.exists():
            if not args.resume or sha256(target) != source_hash:
                raise RuntimeError(f"existing canonical target differs: {target}")
        else:
            shutil.copy2(source, target)
        shutil.copy2(metadata, target_dir / "metadata.json")
        target_hash = sha256(target)
        if target_hash != source_hash:
            raise RuntimeError(f"copy hash mismatch for {method}")
        rows[method] = {"source": str(source.resolve()), "target": str(target.resolve()),
                        "bytes": target.stat().st_size, "sha256": target_hash,
                        "metadata_sha256": sha256(target_dir / "metadata.json")}
    write_json(RUN / "canonical/copy_provenance.json", {
        "operation": "byte-for-byte copy of immutable audited 064 canonical inputs",
        "source_root": str(args.source.resolve()), "states": rows})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
