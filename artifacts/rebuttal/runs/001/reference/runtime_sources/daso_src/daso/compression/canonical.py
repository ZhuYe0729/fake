from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from daso.reporting.provenance import sha256


@dataclass(frozen=True)
class CanonicalState:
    method: str
    path: Path
    sha256: str
    pruning: str
    quantized: bool


def validate_canonical_manifest(path: str | Path) -> list[CanonicalState]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    states: list[CanonicalState] = []
    for method in ("sparse_bf16", "sparse_nvfp4"):
        item = payload["states"][method]
        state_path = Path(item["state"])
        if not state_path.is_absolute():
            state_path = manifest_path.parent / state_path
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        actual = sha256(state_path)
        if actual != item["sha256"]:
            raise ValueError(f"canonical state hash mismatch: {method}")
        pruning = str(item.get("pruning", "sparsegpt"))
        if pruning != "sparsegpt":
            raise ValueError(f"canonical state must use SparseGPT: {method}")
        quantized = bool(item.get("quantized", False))
        if method == "sparse_nvfp4" and quantized:
            raise ValueError("sparse_nvfp4 canonical state must remain BF16 before final export")
        states.append(CanonicalState(method, state_path, actual, pruning, quantized))
    return states

