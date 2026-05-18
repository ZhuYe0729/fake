from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fake.compression.checkpoint import load_checkpoint_into_model
from fake.kernels.cutlass_sparse_bf16 import (
    CutlassSparseBF16Config,
    SparseBF16ReplacementReport,
    replace_linear_with_cutlass_sparse_bf16,
)
from fake.models.dinov3 import (
    DEFAULT_DINOV3_BACKBONE_PATH,
    DEFAULT_DINOV3_HEAD_PATH,
    load_dinov3_vit7b16_dense_classifier,
)


def load_dinov3_vit7b16_cutlass_sparse_bf16_classifier(
    backbone_path: str | Path = DEFAULT_DINOV3_BACKBONE_PATH,
    head_path: str | Path = DEFAULT_DINOV3_HEAD_PATH,
    device: str | torch.device = "cuda",
    sparse_config: CutlassSparseBF16Config | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], SparseBF16ReplacementReport, dict[str, Any]]:
    model, config = load_dinov3_vit7b16_dense_classifier(
        backbone_path=backbone_path,
        head_path=head_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    checkpoint_metadata = load_checkpoint_into_model(model, checkpoint_path)
    model = model.to(dtype=torch.bfloat16)
    report = replace_linear_with_cutlass_sparse_bf16(
        model=model,
        model_name="dinov3_vit7b16",
        config=sparse_config or CutlassSparseBF16Config(),
    )
    model.eval()
    return model, config, report, checkpoint_metadata
