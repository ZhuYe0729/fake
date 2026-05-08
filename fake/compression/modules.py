from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    module: nn.Module
    kind: str
    columns: int
    reason: str


def select_compressible_modules(model: nn.Module, model_name: str) -> list[ModuleInfo]:
    if model_name == "maxvit":
        return _select_maxvit_modules(model)
    if model_name == "dinov3_vit7b16":
        return _select_dinov3_modules(model)
    raise ValueError(f"Unsupported model for compression: {model_name}")


def _select_maxvit_modules(model: nn.Module) -> list[ModuleInfo]:
    modules: list[ModuleInfo] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name.startswith("head"):
                continue
            modules.append(ModuleInfo(name, module, "linear", module.in_features, "maxvit_attention_or_mlp_linear"))
            continue
        if isinstance(module, nn.Conv2d):
            if name.startswith("stem") or ".se." in name or ".shortcut." in name:
                continue
            if module.groups != 1:
                continue
            if tuple(module.kernel_size) != (1, 1):
                continue
            if not (name.endswith("conv1_1x1") or name.endswith("conv3_1x1")):
                continue
            columns = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            modules.append(ModuleInfo(name, module, "conv2d", columns, "maxvit_mbconv_pointwise_conv"))
    return modules


def _select_dinov3_modules(model: nn.Module) -> list[ModuleInfo]:
    suffixes = (
        "attention.k_proj",
        "attention.v_proj",
        "attention.q_proj",
        "attention.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    modules: list[ModuleInfo] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("backbone.layer."):
            continue
        if not name.endswith(suffixes):
            continue
        modules.append(ModuleInfo(name, module, "linear", module.in_features, "dinov3_transformer_projection"))
    return modules


def flatten_weight(module: nn.Module):
    weight = module.weight.data
    if isinstance(module, nn.Conv2d):
        return weight.flatten(1)
    if isinstance(module, nn.Linear):
        return weight
    raise TypeError(f"Unsupported module type: {type(module)}")


def restore_weight_shape(module: nn.Module, matrix):
    return matrix.reshape_as(module.weight.data)

