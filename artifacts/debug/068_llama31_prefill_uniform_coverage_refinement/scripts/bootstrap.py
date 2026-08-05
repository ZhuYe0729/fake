#!/usr/bin/env python3
"""Copy immutable seed policies and record their provenance."""
from __future__ import annotations

import hashlib
import json
import shutil

from scenario import EXP, MODEL, SOURCE_058, SOURCE_060, SOURCE_061_MODEL


SEEDS = {
    "point_004": SOURCE_061_MODEL / "pareto/policies/point_004.json",
    "point_006": SOURCE_061_MODEL / "pareto/policies/point_006.json",
    "point_008": SOURCE_061_MODEL / "pareto/policies/point_008.json",
    "bridge_dense_nvfp4_088": SOURCE_058 / "pareto/policies/bridge_dense_nvfp4_088.json",
    "bridge_dense_nvfp4_104": SOURCE_058 / "pareto/policies/bridge_dense_nvfp4_104.json",
    "bridge_dense_nvfp4_120": SOURCE_058 / "pareto/policies/bridge_dense_nvfp4_120.json",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    policy_dir = EXP / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for policy_id, source in SEEDS.items():
        destination = policy_dir / f"{policy_id}.json"
        shutil.copy2(source, destination)
        records.append({
            "policy_id": policy_id,
            "source": str(source),
            "source_sha256": sha256(source),
            "copied_sha256": sha256(destination),
        })
    uniform = SOURCE_060 / "data/complete_results.csv"
    manifest = {
        "model_path": str(MODEL),
        "uniform_source": str(uniform),
        "uniform_source_sha256": sha256(uniform),
        "speed_protocol": str(SOURCE_061_MODEL.parent / "SPEED_PROTOCOL.md"),
        "seed_policies": records,
    }
    target = EXP / "source_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
