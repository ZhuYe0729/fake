#!/usr/bin/env python3
"""Point copied policy manifests at this experiment's policy files."""

from __future__ import annotations

import hashlib
import json

from scenario import EXP


def main() -> None:
    directory = EXP / "policies/prefill_decode"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest:
        policy = directory / f"{row['policy_id']}.json"
        if not policy.exists():
            raise FileNotFoundError(policy)
        row["path"] = str(policy)
        row["source_policy_sha256"] = hashlib.sha256(policy.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
