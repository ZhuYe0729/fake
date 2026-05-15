from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fake.kernels.flashinfer_nvfp4 import (
    FlashInferNVFP4Config,
    ReplacementReport,
    replace_linear_with_flashinfer_nvfp4,
)
from fake.models.dinov3 import (
    DEFAULT_DINOV3_BACKBONE_PATH,
    DEFAULT_DINOV3_HEAD_PATH,
    load_dinov3_vit7b16_dense_classifier,
)


def load_dinov3_vit7b16_flashinfer_nvfp4_classifier(
    backbone_path: str | Path = DEFAULT_DINOV3_BACKBONE_PATH,
    head_path: str | Path = DEFAULT_DINOV3_HEAD_PATH,
    device: str | torch.device = "cuda",
    dtype: str = "bf16",
    nvfp4_config: FlashInferNVFP4Config | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], ReplacementReport]:
    model, config = load_dinov3_vit7b16_dense_classifier(
        backbone_path=backbone_path,
        head_path=head_path,
        device=device,
    )
    model = model.to(dtype=_resolve_dtype(dtype))
    report = replace_linear_with_flashinfer_nvfp4(
        model=model,
        model_name="dinov3_vit7b16",
        config=nvfp4_config or FlashInferNVFP4Config(),
    )
    return model, config, report


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported DINOv3 NVFP4 dtype: {dtype}")
