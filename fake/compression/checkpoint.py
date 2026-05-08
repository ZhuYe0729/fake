from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    strict: bool = True,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {}
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = payload.get("metadata", {})
    else:
        state_dict = payload
        metadata = {}
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load checkpoint cleanly: missing={missing}, unexpected={unexpected}")
    return metadata


def checkpoint_csv_fields(metadata: dict[str, Any], checkpoint_path: str | None, method: str) -> dict[str, object]:
    return {
        "checkpoint_path": checkpoint_path or "",
        "compression_method": metadata.get("method", method),
        "sparsity": metadata.get("sparsity", ""),
        "nvfp4_group_size": metadata.get("nvfp4_group_size", ""),
        "calib_samples": metadata.get("calib_samples", ""),
    }

