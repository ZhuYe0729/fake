#!/usr/bin/env python3
"""Copy the fixed 72-policy design, with provenance, into the 057 experiment."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/debug/039_llama31_8b_instruct_prefill_decode_pareto/policies/prefill_decode"
EXP = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    target = EXP / "policies/prefill_decode"
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in source_manifest:
        src = SOURCE / f"{row['policy_id']}.json"
        policy = json.loads(src.read_text())
        if len(policy["method_map"]) != 128:
            raise ValueError(f"unexpected module count in {src}")
        dst = target / src.name
        if not dst.exists() or digest(dst) != digest(src):
            shutil.copy2(src, dst)
        manifest.append({"policy_id": row["policy_id"], "split": row["split"],
                         "policy_kind": row["policy_kind"], "path": str(dst),
                         "source": str(src), "source_policy_sha256": digest(src),
                         "policy_sha256": digest(dst)})
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    architecture = {"scenario": "prefill_decode", "batch": 8, "input_tokens": 2048,
        "output_tokens": 64, "m_prefill": 16384, "m_decode": 8, "module_count": 128,
        "model_path": "/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "source_policy_design": str(SOURCE)}
    (EXP / "architecture_manifest.json").write_text(json.dumps(architecture, indent=2) + "\n")
    print(json.dumps({"policies": len(manifest), "target": str(target)}))


if __name__ == "__main__":
    main()
