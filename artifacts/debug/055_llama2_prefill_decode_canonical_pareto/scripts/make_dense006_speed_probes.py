#!/usr/bin/env python3
"""Create single-factor runtime probes from the dense_006 solver policy."""
from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat"
SOURCE = EXP / "pareto/dense_speed/policies/dense_006.json"
OUT = EXP / "pareto/dense_speed_refine/probes"


def write(policy_id: str, changes: dict[str, str]) -> None:
    policy = json.loads(SOURCE.read_text())
    policy["policy_id"] = policy_id
    policy["policy_kind"] = "runtime_single_factor_probe"
    policy["probe_source"] = "dense_006"
    policy["probe_changes"] = changes
    policy["method_map"] = copy.deepcopy(policy["method_map"])
    for module, method in changes.items():
        policy["method_map"][module]["prefill_method"] = method
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{policy_id}.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")


def main() -> None:
    write("probe_down1_sparse_bf16", {
        "model.layers.1.mlp.down_proj": "sparse_bf16",
    })
    write("probe_down31_sparse_bf16", {
        "model.layers.31.mlp.down_proj": "sparse_bf16",
    })
    write("probe_qkv31_sparse_nvfp4", {
        "model.layers.31.self_attn.qkv_proj": "sparse_nvfp4",
    })
    write("probe_downs_sparse_bf16", {
        "model.layers.1.mlp.down_proj": "sparse_bf16",
        "model.layers.31.mlp.down_proj": "sparse_bf16",
    })


if __name__ == "__main__":
    main()
