#!/usr/bin/env python3
"""Create the immutable 054 policy/sample bundle from 053."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/debug/053_llama2_prefill_phase_unified_quality_recalibration/llama2_7b_chat"
TARGET = ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    source_manifest = json.loads((SOURCE / "policies/prefill_only/manifest.json").read_text())
    target_policy_dir = TARGET / "policies/prefill_only"
    target_policy_dir.mkdir(parents=True)
    manifest = []
    for row in source_manifest:
        source = Path(row["path"])
        target = target_policy_dir / source.name
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"policy copy mismatch: {row['policy_id']}")
        manifest.append({**row, "path": str(target)})
    (target_policy_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sample_source = SOURCE / "samples/wikitext_2048_targets.pt"
    sample_target = TARGET / "samples/wikitext_2048_targets.pt"
    sample_target.parent.mkdir(parents=True)
    shutil.copy2(sample_source, sample_target)
    (TARGET / "provenance.json").write_text(json.dumps({
        "source_bundle": str(SOURCE),
        "policy_manifest_sha256": sha256(SOURCE / "policies/prefill_only/manifest.json"),
        "sample_sha256": sha256(sample_target),
        "sparse_source": "canonical_sparsegpt",
        "sparse_nvfp4_quantization": "single_phase_exporter_conversion",
        "compressed_runtime": "phase_hetero_mytest",
    }, indent=2) + "\n")
    print(json.dumps({"target": str(TARGET), "policies": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
