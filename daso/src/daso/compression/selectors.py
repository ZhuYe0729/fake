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


def select_compressible_modules(model: nn.Module, family: str) -> list[ModuleInfo]:
    if family in {"llama", "fakevlm"}:
        return _language_linears(model, multimodal=family == "fakevlm")
    if family == "maxvit":
        return _maxvit(model)
    if family == "dinov3":
        return _suffix_linears(model, ("backbone.layer.",), _DINO_SUFFIXES, "dinov3_projection")
    if family == "mirror":
        return _suffix_linears(model, ("backbone.dino.layer.", "backbone.dino.model.layer."),
                               _MIRROR_SUFFIXES, "mirror_projection")
    raise ValueError(f"unsupported model family: {family}")


_DINO_SUFFIXES = ("attention.k_proj", "attention.v_proj", "attention.q_proj",
                  "attention.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_MIRROR_SUFFIXES = ("attention.k_proj.base_layer", "attention.v_proj.base_layer",
                    "attention.q_proj.base_layer", "attention.o_proj", "mlp.gate_proj",
                    "mlp.up_proj", "mlp.down_proj")


def _language_linears(model: nn.Module, *, multimodal: bool) -> list[ModuleInfo]:
    output = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name == "lm_head" or name.endswith(".lm_head"):
            continue
        if multimodal and any(part in name for part in ("vision_tower", "multi_modal_projector")):
            continue
        output.append(ModuleInfo(name, module, "linear", module.in_features,
                                 "language_linear"))
    return output


def _maxvit(model: nn.Module) -> list[ModuleInfo]:
    output = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.startswith("head"):
            output.append(ModuleInfo(name, module, "linear", module.in_features,
                                     "maxvit_attention_or_mlp_linear"))
        elif isinstance(module, nn.Conv2d) and module.groups == 1 and tuple(module.kernel_size) == (1, 1) \
                and (name.endswith("conv1_1x1") or name.endswith("conv3_1x1")):
            output.append(ModuleInfo(name, module, "conv2d", module.in_channels,
                                     "maxvit_mbconv_pointwise_conv"))
    return output


def _suffix_linears(model: nn.Module, prefixes: tuple[str, ...], suffixes: tuple[str, ...],
                    reason: str) -> list[ModuleInfo]:
    return [ModuleInfo(name, module, "linear", module.in_features, reason)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name.startswith(prefixes) and name.endswith(suffixes)]

