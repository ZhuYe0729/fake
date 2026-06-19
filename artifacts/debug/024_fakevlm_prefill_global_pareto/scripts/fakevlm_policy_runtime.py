#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fake.compression.modules import flatten_weight, restore_weight_shape, select_compressible_modules
from fake.compression.pruning import prune_dense_2_4, prune_nvfp4_pair_2_4
from fake.kernels.cutlass_nvfp4 import _load_cutlass_nvfp4_symbols
from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear, SPARSE_BF16_BLOCKED_SHAPES, _load_cutlass_sparse_bf16_symbols
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear, _load_cutlass_sparse_nvfp4_symbols


@dataclass(frozen=True)
class RuntimeReport:
    replaced_linear_count: int
    skipped_linear_count: int
    backend_counts: dict[str, int]
    skipped: list[dict[str, str]]


def apply_policy_runtime(
    model: nn.Module,
    policy: dict[str, Any],
    *,
    calib_loader: DataLoader | None,
    device: torch.device,
    calib_samples: int,
    input_dtype: torch.dtype = torch.bfloat16,
) -> RuntimeReport:
    modules = policy["modules"]
    selected = {info.name: info for info in select_compressible_modules(model, "fakevlm") if info.kind == "linear"}
    sparse_names = [module_name(item) for item in modules if selected_method(item).startswith("sparse_")]
    hessian = {}
    if sparse_names:
        from eval_fakevlm_uniform_accuracy import collect_vlm_hessian_diag

        if calib_loader is None:
            raise RuntimeError("calib_loader is required for sparse policy runtime")
        sparse_infos = [selected[name] for name in sparse_names if name in selected]
        hessian = collect_vlm_hessian_diag(
            model=model,
            modules=sparse_infos,
            dataloader=calib_loader,
            device=device,
            input_dtype=input_dtype,
            max_samples=calib_samples,
        )

    backend_counts: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []
    replaced = 0
    for item in modules:
        name = module_name(item)
        method = selected_method(item)
        backend_counts[method] += 1
        if method == "dense_bf16":
            continue
        try:
            parent, child_name = resolve_parent(model, name)
            linear = getattr(parent, child_name)
            if not isinstance(linear, nn.Linear):
                skipped.append({"name": name, "reason": f"not_linear:{type(linear).__name__}"})
                continue
            prepared = prepare_linear_for_method(linear, method, hessian.get(name))
            setattr(parent, child_name, make_backend_module(method, prepared).eval())
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return RuntimeReport(replaced, len(skipped), dict(sorted(backend_counts.items())), skipped)


def prepare_linear_for_method(linear: nn.Linear, method: str, hdiag: torch.Tensor | None) -> nn.Linear:
    cloned = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, device=linear.weight.device, dtype=torch.bfloat16)
    cloned.weight.data.copy_(linear.weight.detach().to(dtype=torch.bfloat16))
    if linear.bias is not None:
        cloned.bias.data.copy_(linear.bias.detach().to(dtype=torch.bfloat16))
    cloned.eval()
    cloned.requires_grad_(False)
    if method == "sparse_bf16":
        result = prune_dense_2_4(flatten_weight(cloned), hdiag)
    elif method == "sparse_nvfp4":
        result = prune_nvfp4_pair_2_4(flatten_weight(cloned), hdiag)
    else:
        result = None
    if result is not None and result.mask is not None:
        cloned.weight.data.copy_(restore_weight_shape(cloned, result.weight))
    return cloned


def make_backend_module(method: str, linear: nn.Linear) -> nn.Module:
    if method == "dense_nvfp4":
        nvfp4_cls, can_use = _load_cutlass_nvfp4_symbols()
        if not can_use(1, linear.out_features, linear.in_features, load_extension=False):
            raise ValueError(f"shape_not_supported:dense_nvfp4:{linear.out_features}x{linear.in_features}")
        return nvfp4_cls.from_linear(linear)
    if method == "sparse_bf16":
        if (linear.out_features, linear.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            raise ValueError(f"shape_not_supported:sparse_bf16_blocked:{linear.out_features}x{linear.in_features}")
        sparse_cls, can_use = _load_cutlass_sparse_bf16_symbols()
        if not can_use(linear.out_features, 8, linear.in_features, load_extension=False):
            raise ValueError(f"shape_not_supported:sparse_bf16:{linear.out_features}x{linear.in_features}")
        return PaddedSparseBF16Linear(sparse_cls.from_linear(linear, prune=False), 8)
    if method == "sparse_nvfp4":
        sparse_cls, can_use = _load_cutlass_sparse_nvfp4_symbols()
        if not can_use(linear.out_features, 32, linear.in_features, load_extension=False):
            raise ValueError(f"shape_not_supported:sparse_nvfp4:{linear.out_features}x{linear.in_features}")
        return PaddedSparseNVFP4Linear(sparse_cls.from_linear(linear, prune=False), 32)
    raise ValueError(f"unsupported method: {method}")


def module_name(item: dict[str, Any]) -> str:
    return str(item.get("module_name") or item.get("name"))


def selected_method(item: dict[str, Any]) -> str:
    return str(item.get("selected_method") or item.get("backend"))


def resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
