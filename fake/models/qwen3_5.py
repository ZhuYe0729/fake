from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_QWEN3_5_MODEL_ID = "Qwen/Qwen3.5-0.6B"
DEFAULT_QWEN3_5_ROOT = Path("/home/agent/wja/data/models/Qwen")
DEFAULT_QWEN3_5_VARIANT = "0.8B"
QWEN3_5_VARIANTS = ("0.8B", "2B", "4B", "9B", "27B")


def qwen3_5_model_path(variant: str = DEFAULT_QWEN3_5_VARIANT) -> Path:
    normalized = variant.strip()
    if normalized not in QWEN3_5_VARIANTS:
        raise ValueError(f"Unsupported Qwen3.5 variant: {variant}")
    return DEFAULT_QWEN3_5_ROOT / f"Qwen3.5-{normalized}"


def load_qwen3_5_dense(
    model_id: str | Path = qwen3_5_model_path(),
    device: str | torch.device = "cpu",
    torch_dtype: str | torch.dtype = "auto",
) -> tuple[nn.Module, dict[str, Any]]:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(model_id),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    config = {
        "model": "Qwen3.5",
        "model_id": str(model_id),
    }
    return model, config
