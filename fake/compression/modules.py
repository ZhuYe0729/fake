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
    if model_name == "mirror":
        return _select_mirror_modules(model)
    if model_name == "qwen3_5":
        return _select_qwen3_5_modules(model)
    if model_name == "llama":
        return _select_llama_modules(model)
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


def _select_mirror_modules(model: nn.Module) -> list[ModuleInfo]:
    suffixes = (
        "attention.k_proj.base_layer",
        "attention.v_proj.base_layer",
        "attention.q_proj.base_layer",
        "attention.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    modules: list[ModuleInfo] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("backbone.dino.layer."):
            continue
        if not name.endswith(suffixes):
            continue
        modules.append(ModuleInfo(name, module, "linear", module.in_features, "mirror_dinov3_transformer_projection"))
    return modules


def _select_qwen3_5_modules(model: nn.Module) -> list[ModuleInfo]:
    modules: list[ModuleInfo] = []
    language_prefixes = _qwen3_5_language_prefixes(model)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        if "" in language_prefixes:
            in_language_model = bool(name)
        else:
            in_language_model = any(name == prefix or name.startswith(f"{prefix}.") for prefix in language_prefixes)
        if not in_language_model:
            continue
        modules.append(ModuleInfo(name, module, "linear", module.in_features, "qwen3_5_language_linear"))
    return modules


def _select_llama_modules(model: nn.Module) -> list[ModuleInfo]:
    modules: list[ModuleInfo] = []
    prefixes = _llama_language_prefixes(model)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        if "" in prefixes:
            in_model = bool(name)
        else:
            in_model = any(name == p or name.startswith(f"{p}.") for p in prefixes)
        if not in_model:
            continue
        modules.append(ModuleInfo(name, module, "linear", module.in_features, "llama_language_linear"))
    return modules


def _llama_language_prefixes(model: nn.Module) -> tuple[str, ...]:
    prefixes: list[str] = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        prefixes.append("model")
    if hasattr(model, "layers"):
        prefixes.append("")
    return tuple(prefixes)


def _qwen3_5_language_prefixes(model: nn.Module) -> tuple[str, ...]:
    prefixes: list[str] = []
    if hasattr(model, "language_model"):
        prefixes.append("language_model")
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "language_model"):
            prefixes.append("model.language_model")
        if hasattr(inner, "layers"):
            prefixes.append("model")
    if hasattr(model, "layers"):
        prefixes.append("")
    return tuple(prefixes)


def flatten_weight(module: nn.Module):
    weight = module.weight.data
    if isinstance(module, nn.Conv2d):
        return weight.flatten(1)
    if isinstance(module, nn.Linear):
        return weight
    raise TypeError(f"Unsupported module type: {type(module)}")


def restore_weight_shape(module: nn.Module, matrix):
    return matrix.reshape_as(module.weight.data)
