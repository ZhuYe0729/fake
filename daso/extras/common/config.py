from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if "model" not in config or "dataset" not in config:
        raise ValueError("extra-model config requires [model] and [dataset] sections")
    return config


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

