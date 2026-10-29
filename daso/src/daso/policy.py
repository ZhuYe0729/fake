from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = (
    "dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours",
)
PHASES = ("prefill", "decode")


def validate_policy(policy: dict[str, Any], *, layers: int | None = None) -> None:
    required = {"policy_id", "scenario", "method_map"}
    missing = required - policy.keys()
    if missing:
        raise ValueError(f"policy missing fields: {sorted(missing)}")
    if policy["scenario"] not in {"prefill_only", "prefill_decode"}:
        raise ValueError(f"invalid scenario: {policy['scenario']}")
    method_map = policy["method_map"]
    if not isinstance(method_map, dict) or not method_map:
        raise ValueError("method_map must be a non-empty object")
    seen_layers: set[int] = set()
    for name, action in method_map.items():
        parts = name.split(".")
        if len(parts) < 4 or not parts[2].isdigit():
            raise ValueError(f"cannot determine layer from module name: {name}")
        layer = int(parts[2])
        seen_layers.add(layer)
        for phase in PHASES:
            key = f"{phase}_method"
            if key not in action:
                raise ValueError(f"{name} missing {key}")
            if action[key] not in METHODS:
                raise ValueError(f"{name} has unsupported {key}={action[key]}")
    if layers is not None and (min(seen_layers) < 0 or max(seen_layers) >= layers):
        raise ValueError(f"policy contains layer outside [0, {layers})")


def load_policy(path: str | Path, *, layers: int | None = None) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text())
    validate_policy(policy, layers=layers)
    return policy


def policy_sha256(policy: dict[str, Any]) -> str:
    validate_policy(policy)
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def uniform_policy(policy_id: str, module_names: list[str], method: str,
                   *, scenario: str = "prefill_decode") -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(method)
    policy = {
        "policy_id": policy_id,
        "scenario": scenario,
        "default_prefill_method": "dense_bf16",
        "default_decode_method": "dense_bf16",
        "modules_to_not_convert": ["lm_head"],
        "method_map": {name: {"prefill_method": method, "decode_method": method}
                       for name in module_names},
    }
    validate_policy(policy)
    return policy

