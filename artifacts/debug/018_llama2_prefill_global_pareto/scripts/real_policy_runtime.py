from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RealPolicyRuntimeReport:
    replaced_linear_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]
    backend_counts: dict[str, int]


def apply_real_policy_runtime(
    model: nn.Module,
    policy: dict[str, Any],
    *,
    prepared_root: Path,
    activation_dtype: torch.dtype,
) -> RealPolicyRuntimeReport:
    states: dict[str, dict[str, torch.Tensor]] = {}
    skipped: list[dict[str, str]] = []
    backend_counts: dict[str, int] = {}
    replaced = 0
    for item in policy["modules"]:
        module_name = item["module_name"]
        method = item["selected_prefill_backend"]
        if method == "dense_bf16":
            backend_counts[method] = backend_counts.get(method, 0) + 1
            continue
        try:
            if method not in states:
                states[method] = load_prepared_state(prepared_root, method)
            parent, child_name = resolve_parent(model, module_name)
            linear = getattr(parent, child_name)
            if not isinstance(linear, nn.Linear):
                skipped.append({"name": module_name, "reason": f"not_linear:{type(linear).__name__}"})
                continue
            copy_prepared_linear_(linear, states[method], module_name)
            setattr(parent, child_name, build_backend_from_prepared_linear(linear, method, activation_dtype=activation_dtype))
            backend_counts[method] = backend_counts.get(method, 0) + 1
            replaced += 1
        except Exception as exc:
            skipped.append({"name": module_name, "reason": f"{type(exc).__name__}:{exc}"})
    return RealPolicyRuntimeReport(
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
        backend_counts=backend_counts,
    )


def load_prepared_state(prepared_root: Path, method: str) -> dict[str, torch.Tensor]:
    path = prepared_root / method / "model.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("method") != method:
        raise RuntimeError(f"prepared artifact method mismatch: expected={method} got={metadata.get('method')}")
    return payload["state_dict"]


def copy_prepared_linear_(linear: nn.Linear, state: dict[str, torch.Tensor], module_name: str) -> None:
    weight_key = f"{module_name}.weight"
    if weight_key not in state:
        raise KeyError(weight_key)
    linear.weight.data.copy_(state[weight_key].to(device=linear.weight.device, dtype=linear.weight.dtype))
    bias_key = f"{module_name}.bias"
    if linear.bias is not None and bias_key in state:
        linear.bias.data.copy_(state[bias_key].to(device=linear.bias.device, dtype=linear.bias.dtype))


def build_backend_from_prepared_linear(
    linear: nn.Linear,
    method: str,
    *,
    activation_dtype: torch.dtype,
) -> nn.Module:
    wrapper = load_wrapper()
    if method == "dense_nvfp4":
        if not wrapper.can_use_cutlass_nvfp4(1, linear.out_features, linear.in_features, load_extension=False):
            raise ValueError(shape_reason(linear, method))
        return wrapper.NVFP4Linear.from_linear(linear)
    if method == "marlin_nvfp4":
        if not wrapper.can_use_marlin_nvfp4(1, linear.out_features, linear.in_features, load_extension=False):
            raise ValueError(shape_reason(linear, method))
        return wrapper.MarlinNVFP4Linear.from_linear(linear, activation_dtype=activation_dtype)
    if method == "sparse_bf16":
        from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear, SPARSE_BF16_BLOCKED_SHAPES, _load_cutlass_sparse_bf16_symbols

        if (linear.out_features, linear.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            raise ValueError(shape_reason(linear, "sparse_bf16_blocked"))
        sparse_cls, can_use = _load_cutlass_sparse_bf16_symbols()
        if not can_use(linear.out_features, 8, linear.in_features, load_extension=False):
            raise ValueError(shape_reason(linear, method))
        return PaddedSparseBF16Linear(sparse_cls.from_linear(linear, prune=False), 8)
    if method == "sparse_nvfp4":
        from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear, _load_cutlass_sparse_nvfp4_symbols

        sparse_cls, can_use = _load_cutlass_sparse_nvfp4_symbols()
        if not can_use(linear.out_features, 32, linear.in_features, load_extension=False):
            raise ValueError(shape_reason(linear, method))
        return PaddedSparseNVFP4Linear(sparse_cls.from_linear(linear, prune=False), 32)
    raise ValueError(f"unsupported real policy backend: {method}")


def load_wrapper():
    from fake.models.qwen3_5_kernels import _load_wrapper

    return _load_wrapper()


def resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def shape_reason(linear: nn.Linear, backend: str) -> str:
    return f"shape_not_supported:{backend}:in_features={linear.in_features},out_features={linear.out_features}"
