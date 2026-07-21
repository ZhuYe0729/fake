#!/usr/bin/env python3
"""Small shared helpers for the 063 reproducibility bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


BUNDLE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("COSPAQ_REPO_ROOT", BUNDLE.parents[2])).resolve()
SRC054 = REPO / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration"
SRC056 = REPO / "artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto"
SRC060 = REPO / "artifacts/debug/060_two_model_two_scenario_result_consolidation"


def env_path(name: str, *, required: bool = True) -> Path | None:
    value = os.environ.get(name)
    if not value:
        if required:
            raise SystemExit(f"missing required environment variable: {name}")
        return None
    return Path(value).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

