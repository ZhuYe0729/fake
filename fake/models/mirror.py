from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
MIRROR_ROOT = REPO_ROOT / "third_party" / "MIRROR"
DEFAULT_MIRROR_WEIGHT_ROOT = Path("/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight")
DEFAULT_MIRROR_MODEL_PATH = DEFAULT_MIRROR_WEIGHT_ROOT / "checkpoint-h-cur.pth"
DEFAULT_MIRROR_MEMORY_PATH = DEFAULT_MIRROR_WEIGHT_ROOT / "mirror_phase1.pth"
DEFAULT_MIRROR_BACKBONE_PATH = DEFAULT_MIRROR_WEIGHT_ROOT / "dinov3-huge"


def load_mirror_dense_detector(
    model_path: str | Path = DEFAULT_MIRROR_MODEL_PATH,
    memory_path: str | Path = DEFAULT_MIRROR_MEMORY_PATH,
    backbone_path: str | Path = DEFAULT_MIRROR_BACKBONE_PATH,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    if str(MIRROR_ROOT) not in sys.path:
        sys.path.insert(0, str(MIRROR_ROOT))
    from models.mirror import build_mirror

    model = build_mirror(memory_path=str(memory_path), backbone_path=str(backbone_path))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    config = {
        "model": "MIRROR-DINOv3-Huge",
        "model_path": str(model_path),
        "memory_path": str(memory_path),
        "backbone_path": str(backbone_path),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }
    return model, config


def load_mirror_compressed_detector(
    checkpoint_path: str | Path,
    model_path: str | Path = DEFAULT_MIRROR_MODEL_PATH,
    memory_path: str | Path = DEFAULT_MIRROR_MEMORY_PATH,
    backbone_path: str | Path = DEFAULT_MIRROR_BACKBONE_PATH,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype | None = None,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    model, config = load_mirror_dense_detector(
        model_path=model_path,
        memory_path=memory_path,
        backbone_path=backbone_path,
        device="cpu",
        torch_dtype=torch_dtype,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load MIRROR compressed checkpoint: missing={missing}, unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    return model, config, metadata


def model_input_dtype(model: nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    return torch.float32
