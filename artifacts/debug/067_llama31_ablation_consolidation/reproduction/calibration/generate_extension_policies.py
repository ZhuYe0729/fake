#!/usr/bin/env python3
"""Generate 18 fixed, non-duplicate calibration-only policies from the 039 design family."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
COUNT = 18


def method_map(index: int) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for layer in range(32):
        bucket = layer // 8
        for module_type in TYPES:
            if index < 5:
                method = METHODS[index]
            elif index < 21:
                cell = index - 5
                selected_bucket, selected_type = cell // 4, TYPES[cell % 4]
                method = METHODS[1 + (cell % 4)] if (bucket, module_type) == (selected_bucket, selected_type) else "dense_bf16"
            elif index < 37:
                bucket_to_change, method_index = divmod(index - 21, 4)
                method = METHODS[1 + method_index] if bucket == bucket_to_change else "dense_bf16"
            else:
                severity = (index - 37) % 5
                probabilities = (
                    (0.75, 0.10, 0.03, 0.02, 0.10),
                    (0.55, 0.18, 0.07, 0.04, 0.16),
                    (0.38, 0.25, 0.09, 0.06, 0.22),
                    (0.22, 0.30, 0.13, 0.10, 0.25),
                    (0.08, 0.30, 0.18, 0.16, 0.28),
                )[severity]
                rng = random.Random(1307 + index * 100003 + layer * 97 + TYPES.index(module_type))
                draw, total = rng.random(), 0.0
                method = METHODS[-1]
                for candidate, probability in zip(METHODS, probabilities):
                    total += probability
                    if draw <= total:
                        method = candidate
                        break
            prefix = "self_attn" if module_type in {"qkv_proj", "o_proj"} else "mlp"
            result[f"model.layers.{layer}.{prefix}.{module_type}"] = {
                "prefill_method": method,
                "decode_method": method,
            }
    return result


def signature(policy_map: dict[str, dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(policy_map, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    source_manifest = json.loads((SOURCE / "policies/prefill_decode/manifest.json").read_text())
    seen = {
        signature(json.loads(Path(row["path"]).read_text())["method_map"]): row["policy_id"]
        for row in source_manifest
    }
    output = OUT / "extension/policies"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for offset in range(COUNT):
        source_index = 72 + offset
        prefill = method_map(source_index)
        decode = method_map((source_index * 17 + 11) % 72)
        for name, item in prefill.items():
            item["decode_method"] = decode[name]["prefill_method"]
        digest = signature(prefill)
        if digest in seen:
            raise RuntimeError(f"extension index {source_index} duplicates {seen[digest]}")
        policy_id = f"e{offset:02d}"
        seen[digest] = policy_id
        policy = {
            "policy_id": policy_id,
            "scenario": "prefill_decode",
            "policy_kind": "calibration_size_extension",
            "default_prefill_method": "dense_bf16",
            "default_decode_method": "dense_bf16",
            "modules_to_not_convert": ["lm_head"],
            "method_map": prefill,
        }
        path = output / f"{policy_id}.json"
        path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        manifest.append({
            "policy_id": policy_id,
            "policy_kind": policy["policy_kind"],
            "source_index": source_index,
            "path": str(path),
            "policy_signature": digest,
        })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    audit = {
        "generator": "exact 039 balanced-mixed family continued at source indices 72..89",
        "policies": len(manifest),
        "existing_policies_checked": len(source_manifest),
        "unique_signatures": len({row["policy_signature"] for row in manifest}),
        "duplicates_with_057": 0,
    }
    (OUT / "extension/policy_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
