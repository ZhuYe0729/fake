from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class INT4Config:
    group_size: int = 32
    scale_precision: str = "fp16"


@dataclass
class INT4Result:
    weight: torch.Tensor
    scales: torch.Tensor | None
    stats: dict[str, object]


def fake_quantize_int4_weight(matrix: torch.Tensor, config: INT4Config) -> INT4Result:
    if config.group_size <= 0:
        return INT4Result(matrix, None, {"status": "skipped", "reason": "invalid_group_size"})
    if matrix.shape[-1] % config.group_size != 0:
        return INT4Result(
            matrix,
            None,
            {
                "status": "skipped",
                "reason": "columns_not_divisible_by_group_size",
                "columns": matrix.shape[-1],
                "group_size": config.group_size,
            },
        )

    original_dtype = matrix.dtype
    grouped = matrix.detach().float().reshape(matrix.shape[0], -1, config.group_size)
    scales = calculate_int4_group_scales(matrix, config).to(device=matrix.device, dtype=torch.float32).unsqueeze(-1)
    q = torch.clamp(torch.round(grouped / scales), -8, 7)
    dequant = (q * scales).reshape_as(matrix).to(original_dtype)
    scale_tensor = _cast_scale_precision(scales.squeeze(-1), config.scale_precision)
    stats = int4_scale_stats(scale_tensor, config)
    return INT4Result(dequant, scale_tensor.cpu(), stats)


def fake_quantize_int4_activation(x: torch.Tensor, config: INT4Config) -> torch.Tensor:
    if config.group_size <= 0:
        raise ValueError(f"Unsupported activation INT4 group_size: {config.group_size}")
    if x.shape[-1] % config.group_size != 0:
        raise ValueError(
            f"Activation last dimension must be divisible by group_size: columns={x.shape[-1]} "
            f"group_size={config.group_size}"
        )
    original_dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], -1, config.group_size)
    # Keep dynamic activation scales in fp32 for the fake op. Casting tiny all-zero
    # groups to fp16 can underflow the clamp value to 0 and produce NaNs.
    scales = (grouped.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-12)
    q = torch.clamp(torch.round(grouped / scales), -8, 7)
    dequant = q * scales
    return dequant.reshape_as(x_float).to(original_dtype)


def calculate_int4_group_scales(matrix: torch.Tensor, config: INT4Config) -> torch.Tensor:
    grouped = matrix.detach().float().reshape(matrix.shape[0], -1, config.group_size)
    scales = (grouped.abs().amax(dim=-1) / 7.0).clamp(min=1e-12)
    return _cast_scale_precision(scales, config.scale_precision)


def int4_scale_stats(scales: torch.Tensor, config: INT4Config) -> dict[str, object]:
    scale_float = scales.float()
    return {
        "status": "ok",
        "format": "signed_symmetric",
        "bits": 4,
        "qmin": -8,
        "qmax": 7,
        "group_size": config.group_size,
        "scale_precision": config.scale_precision,
        "global_scale": False,
        "zero_point": False,
        "num_groups": int(scales.numel()),
        "scale_min": float(scale_float.min().item()),
        "scale_max": float(scale_float.max().item()),
    }


def quantize_int4_column_grouped(
    column: torch.Tensor,
    scales: torch.Tensor,
    column_idx: int,
    config: INT4Config,
) -> torch.Tensor:
    group_start = (column_idx // config.group_size) * config.group_size
    if group_start // config.group_size >= scales.shape[1]:
        raise ValueError(
            f"INT4 group exceeds matrix columns: column={column_idx} "
            f"group_size={config.group_size} groups={scales.shape[1]}"
        )
    scale = scales[:, group_start // config.group_size].to(device=column.device, dtype=torch.float32)
    q = torch.clamp(torch.round(column.float() / scale), -8, 7)
    return (q * scale).to(column.dtype)


def _cast_scale_precision(scales: torch.Tensor, scale_precision: str) -> torch.Tensor:
    if scale_precision == "fp16":
        return scales.to(torch.float16)
    if scale_precision == "bf16":
        return scales.to(torch.bfloat16)
    if scale_precision == "fp32":
        return scales.to(torch.float32)
    raise ValueError(f"Unsupported INT4 scale precision: {scale_precision}")
