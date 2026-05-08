from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_DINOV3_BACKBONE_PATH = Path(
    "/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m"
)
DEFAULT_DINOV3_HEAD_PATH = Path(
    "/data/home/scxj523/run/wja/data/models/facebook/"
    "dinov3_vit7b16_imagenet1k_linear_head/"
    "dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth"
)


class DINOv3LinearClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, linear_head: nn.Module, num_register_tokens: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.num_register_tokens = num_register_tokens

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1 + self.num_register_tokens :]
        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return self.linear_head(linear_input)


def load_dinov3_vit7b16_dense_classifier(
    backbone_path: str | Path = DEFAULT_DINOV3_BACKBONE_PATH,
    head_path: str | Path = DEFAULT_DINOV3_HEAD_PATH,
    device: str | torch.device = "cuda",
) -> tuple[nn.Module, dict[str, Any]]:
    from transformers import AutoModel

    backbone_dir = Path(backbone_path)
    config = _load_config(backbone_dir)
    backbone = AutoModel.from_pretrained(
        str(backbone_dir),
        local_files_only=True,
        torch_dtype="auto",
        trust_remote_code=False,
    )
    runtime_dtype = model_input_dtype(backbone)
    head = nn.Linear(2 * int(config["hidden_size"]), 1000)
    state_dict = torch.load(Path(head_path), map_location="cpu")
    head.load_state_dict(state_dict, strict=True)
    head = head.to(dtype=runtime_dtype)

    model = DINOv3LinearClassifier(
        backbone=backbone,
        linear_head=head,
        num_register_tokens=int(config.get("num_register_tokens", 4)),
    )
    model = model.to(device)
    model.eval()
    return model, config


def model_input_dtype(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _load_config(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "config.json").open("r") as f:
        return json.load(f)

