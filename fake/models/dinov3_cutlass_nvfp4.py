from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fake.kernels.cutlass_nvfp4 import (
    CutlassNVFP4Config,
    ReplacementReport,
    replace_linear_with_cutlass_nvfp4,
)
from fake.models.dinov3 import (
    DEFAULT_DINOV3_BACKBONE_PATH,
    DEFAULT_DINOV3_HEAD_PATH,
    load_dinov3_vit7b16_dense_classifier,
)


def load_dinov3_vit7b16_cutlass_nvfp4_classifier(
    backbone_path: str | Path = DEFAULT_DINOV3_BACKBONE_PATH,
    head_path: str | Path = DEFAULT_DINOV3_HEAD_PATH,
    device: str | torch.device = "cuda",
    nvfp4_config: CutlassNVFP4Config | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], ReplacementReport]:
    model, config = load_dinov3_vit7b16_dense_classifier(
        backbone_path=backbone_path,
        head_path=head_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(dtype=torch.bfloat16)
    report = replace_linear_with_cutlass_nvfp4(
        model=model,
        model_name="dinov3_vit7b16",
        config=nvfp4_config or CutlassNVFP4Config(),
    )
    model.eval()
    return model, config, report
