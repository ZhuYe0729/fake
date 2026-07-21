#!/usr/bin/env python3
"""Freeze the prefill-only design without reusing invalid old NLL labels."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from scenario import EXP, MODEL, SOURCE_038, SOURCE_057, CANONICAL, LOCAL_ERRORS, BATCH, INPUT_TOKENS, METHODS

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def main() -> None:
    policy_src = SOURCE_038 / "policies/prefill_only"
    sample_src = SOURCE_038 / "samples/wikitext_2048.pt"
    required = [MODEL, policy_src / "manifest.json", sample_src,
                CANONICAL / "sparse_bf16/model.pt", CANONICAL / "sparse_nvfp4/model.pt",
                *[LOCAL_ERRORS / f"prefill_{m}.csv" for m in METHODS[1:]]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing reusable assets: " + ", ".join(missing))
    target = EXP / "policies/prefill_only"; target.mkdir(parents=True, exist_ok=True)
    for source in sorted(policy_src.glob("p*.json")):
        shutil.copy2(source, target / source.name)
    manifest = json.loads((policy_src / "manifest.json").read_text())
    for row in manifest:
        row["path"] = str(target / f"{row['policy_id']}.json")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sample_target = EXP / "samples/wikitext_2048.pt"; sample_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_src, sample_target)
    source_manifest = json.loads((SOURCE_057 / "canonical/canonical_manifest.json").read_text())
    provenance = {"scenario": "prefill_only", "batch": BATCH, "input_tokens": INPUT_TOKENS,
                  "model": str(MODEL), "policy_source": str(policy_src), "sample_source": str(sample_src),
                  "sample_sha256": digest(sample_target), "canonical_source": str(SOURCE_057),
                  "canonical_manifest": source_manifest,
                  "local_error_source": str(LOCAL_ERRORS),
                  "forbidden_export_flag": "--prune"}
    (EXP / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"policies": len(manifest), "sample": str(sample_target), "experiment": str(EXP)}, indent=2))
if __name__ == "__main__": main()
