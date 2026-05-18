from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fake.compression.modules import select_compressible_modules
from fake.compression.nvfp4 import NVFP4Config, fake_quantize_nvfp4_activation


@dataclass(frozen=True)
class ActivationQuantConfig:
    group_size: int = 16
    scale_rule: str = "four_over_six_mse"
    scale_precision: str = "fp16"


class ActivationFakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, config: ActivationQuantConfig) -> None:
        super().__init__()
        self.linear = linear
        self.config = config
        self.nvfp4_config = NVFP4Config(
            group_size=config.group_size,
            scale_precision=config.scale_precision,
            scale_rule=config.scale_rule,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_quant = fake_quantize_nvfp4_activation(x, self.nvfp4_config)
        return F.linear(x_quant, self.linear.weight, self.linear.bias)


class ActivationFakeQuantConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, config: ActivationQuantConfig) -> None:
        super().__init__()
        self.conv = conv
        self.config = config
        self.nvfp4_config = NVFP4Config(
            group_size=config.group_size,
            scale_precision=config.scale_precision,
            scale_rule=config.scale_rule,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channels_last = x.permute(0, 2, 3, 1).contiguous()
        x_quant = fake_quantize_nvfp4_activation(x_channels_last, self.nvfp4_config)
        x_quant = x_quant.permute(0, 3, 1, 2).contiguous()
        return self.conv(x_quant)


def apply_dinov3_activation_fake_quant(
    model: nn.Module,
    group_size: int,
    scale_rule: str = "four_over_six_mse",
    scale_precision: str = "fp16",
) -> int:
    return apply_activation_fake_quant(
        model,
        model_name="dinov3_vit7b16",
        group_size=group_size,
        scale_rule=scale_rule,
        scale_precision=scale_precision,
    )


def apply_activation_fake_quant(
    model: nn.Module,
    model_name: str,
    group_size: int,
    scale_rule: str = "four_over_six_mse",
    scale_precision: str = "fp16",
) -> int:
    config = ActivationQuantConfig(group_size=group_size, scale_rule=scale_rule, scale_precision=scale_precision)
    replaced = 0
    for info in select_compressible_modules(model, model_name):
        parent, child_name = _resolve_parent(model, info.name)
        child = getattr(parent, child_name)
        if isinstance(child, (ActivationFakeQuantLinear, ActivationFakeQuantConv2d)):
            continue
        if isinstance(child, nn.Linear):
            setattr(parent, child_name, ActivationFakeQuantLinear(child, config))
        elif isinstance(child, nn.Conv2d):
            setattr(parent, child_name, ActivationFakeQuantConv2d(child, config))
        else:
            raise TypeError(f"Expected nn.Linear or nn.Conv2d at {info.name}, got {type(child).__name__}")
        replaced += 1
    return replaced


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
