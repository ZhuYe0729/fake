#!/usr/bin/env python3
"""Create the immutable input bundle for canonical Llama2 prefill-decode."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/prefill_decode/pareto/policies"
SAMPLES = ROOT / "artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/samples/wikitext_2048_80.pt"
OUT = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat"
CANONICAL = ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/canonical/prepared"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    policies = OUT / "seed_policies"
    policies.mkdir(parents=True)
    for source in sorted(OLD.glob("point_*.json")):
        shutil.copy2(source, policies / source.name)
    target_sample = OUT / "samples/wikitext_2048_80.pt"
    target_sample.parent.mkdir(parents=True)
    shutil.copy2(SAMPLES, target_sample)

    smoke = json.loads((policies / "point_010.json").read_text())
    # The policy must exercise distinct phase maps plus both canonical sparse
    # sources; it is a runtime gate, not a solver candidate.
    smoke["method_map"]["model.layers.0.mlp.down_proj"] = {
        "prefill_method": "sparse_bf16", "decode_method": "sparse_nvfp4"}
    smoke["method_map"]["model.layers.1.mlp.gate_up_proj"] = {
        "prefill_method": "sparse_nvfp4", "decode_method": "sparse_bf16"}
    smoke_path = OUT / "smoke/policy.json"
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_text(json.dumps(smoke, indent=2) + "\n")

    payload = {"old_policy_source": str(OLD), "old_policy_count": len(list(OLD.glob("point_*.json"))),
               "sample_source": str(SAMPLES), "sample_sha256": sha256(target_sample),
               "canonical_sparse_bf16_state": str(CANONICAL / "sparse_bf16/model.pt"),
               "canonical_sparse_nvfp4_state": str(CANONICAL / "sparse_nvfp4/model.pt"),
               "canonical_mode": "required_no_direct_prune", "runtime": "phase_hetero_mytest"}
    (OUT / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "sample": str(target_sample), "smoke": str(smoke_path)}, indent=2))


if __name__ == "__main__":
    main()
