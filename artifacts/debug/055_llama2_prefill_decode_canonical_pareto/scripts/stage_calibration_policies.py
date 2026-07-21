#!/usr/bin/env python3
"""Stage the 72 controlled dual-phase policy *designs* for canonical relabeling."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/policies/prefill_decode"
OUT = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat/policies/prefill_decode"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    OUT.mkdir(parents=True)
    manifest = []
    for row in source_manifest:
        source = Path(row["path"])
        target = OUT / source.name
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"copy mismatch: {row['policy_id']}")
        manifest.append({**row, "path": str(target), "source_policy_sha256": digest(source)})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"policies": len(manifest), "train": sum(row['split'] == 'train' for row in manifest),
                      "holdout": sum(row['split'] == 'holdout' for row in manifest)}, indent=2))


if __name__ == "__main__":
    main()
