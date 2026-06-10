from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_LLAMA_ROOT = Path("/home/agent/wja/data/models/LLM-Research")
LLAMA2_MODEL_ID = str(DEFAULT_LLAMA_ROOT / "llama-2-7b")
LLAMA31_MODEL_ID = str(DEFAULT_LLAMA_ROOT / "Meta-Llama-3.1-8B-Instruct")


def load_llama2_dense(
    model_id: str | Path = LLAMA2_MODEL_ID,
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
        "model": "Llama-2-7B",
        "model_id": str(model_id),
    }
    return model, config


def load_llama31_dense(
    model_id: str | Path = LLAMA31_MODEL_ID,
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
        "model": "Llama-3.1-8B-Instruct",
        "model_id": str(model_id),
    }
    return model, config
