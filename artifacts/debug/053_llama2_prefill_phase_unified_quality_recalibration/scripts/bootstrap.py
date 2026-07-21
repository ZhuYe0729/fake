#!/usr/bin/env python3
"""Copy the immutable 046 Llama2 policy and sample design into 053."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat"
TARGET = ROOT / "artifacts/debug/053_llama2_prefill_phase_unified_quality_recalibration/llama2_7b_chat"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_manifest = json.loads((SOURCE / "policies/prefill_only/manifest.json").read_text())
    sample_source = SOURCE / "samples/wikitext_2048_targets.pt"
    sample_target = TARGET / "samples/wikitext_2048_targets.pt"
    if TARGET.exists():
        raise FileExistsError(f"053 target already exists: {TARGET}")
    sample_target.parent.mkdir(parents=True)
    shutil.copy2(sample_source, sample_target)
    policy_dir = TARGET / "policies/prefill_only"
    policy_dir.mkdir(parents=True)
    manifest = []
    for row in source_manifest:
        source = Path(row["path"])
        target = policy_dir / source.name
        shutil.copy2(source, target)
        if sha256(target) != row["sha256"]:
            raise RuntimeError(f"policy hash changed: {row['policy_id']}")
        manifest.append({**row, "path": str(target)})
    (policy_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = json.loads((SOURCE / "samples/metadata.json").read_text())
    metadata.update({"copied_from": str(SOURCE), "source_sample_sha256": sha256(sample_source), "tensor_sha256": sha256(sample_target), "experiment": "053_phase_unified"})
    (TARGET / "samples/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (TARGET / "provenance.json").write_text(json.dumps({"source_bundle": str(SOURCE), "source_manifest_sha256": sha256(SOURCE / "policies/prefill_only/manifest.json"), "sample_sha256": sha256(sample_target), "compressed_runtime": "phase_hetero_mytest", "dense_reference": "raw_hf_bf16"}, indent=2) + "\n")
    print(json.dumps({"target": str(TARGET), "policies": len(manifest), "sample_sha256": sha256(sample_target)}, indent=2))


if __name__ == "__main__":
    main()
