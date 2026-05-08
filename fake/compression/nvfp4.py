from __future__ import annotations

from dataclasses import dataclass

import torch


FP4_E2M1_MAX = 6.0


@dataclass(frozen=True)
class NVFP4Config:
    group_size: int = 16
    scale_precision: str = "fp16"
    scale_remap: str = "none"
    scale_factor: float = 1.0
    scale_remap_gamma: float = 1.0
    scale_remap_mu: float = 5.0


@dataclass
class NVFP4Result:
    weight: torch.Tensor
    scales: torch.Tensor | None
    stats: dict[str, object]


def fake_quantize_nvfp4_weight(matrix: torch.Tensor, config: NVFP4Config) -> NVFP4Result:
    if config.group_size not in (16, 32):
        return NVFP4Result(matrix, None, {"status": "skipped", "reason": "invalid_group_size"})
    if matrix.shape[-1] % config.group_size != 0:
        return NVFP4Result(
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
    x = matrix.detach().float()
    grouped = x.reshape(x.shape[0], -1, config.group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True) / FP4_E2M1_MAX
    scales = scales.clamp(min=1e-12)

    normalized = grouped / scales
    normalized = _apply_remap(normalized, config)
    q = cast_to_fp4(normalized)
    dequant = _undo_remap(q, config) * scales
    dequant = dequant.reshape_as(x).to(original_dtype)
    scale_tensor = _cast_scale_precision(scales.squeeze(-1), config.scale_precision)
    stats = {
        "status": "ok",
        "group_size": config.group_size,
        "scale_precision": config.scale_precision,
        "scale_remap": config.scale_remap,
        "scale_factor": config.scale_factor,
        "num_groups": int(scale_tensor.numel()),
        "scale_min": float(scales.min().item()),
        "scale_max": float(scales.max().item()),
    }
    return NVFP4Result(dequant, scale_tensor.cpu(), stats)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    y = torch.abs(x)
    out = torch.empty_like(y)
    out[(y >= 0.0) & (y <= 0.25)] = 0.0
    out[(y > 0.25) & (y < 0.75)] = 0.5
    out[(y >= 0.75) & (y <= 1.25)] = 1.0
    out[(y > 1.25) & (y < 1.75)] = 1.5
    out[(y >= 1.75) & (y <= 2.5)] = 2.0
    out[(y > 2.5) & (y < 3.5)] = 3.0
    out[(y >= 3.5) & (y <= 5.0)] = 4.0
    out[y > 5.0] = 6.0
    return out * sign


def _apply_remap(x: torch.Tensor, config: NVFP4Config) -> torch.Tensor:
    if config.scale_remap == "none":
        return x
    if config.scale_remap == "linear":
        return x * config.scale_factor
    if config.scale_remap in ("power", "auto"):
        return torch.sign(x) * 6.0 * (torch.abs(x) / 6.0).pow(config.scale_remap_gamma)
    if config.scale_remap == "mulaw":
        import math

        return torch.sign(x) * 6.0 * torch.log1p(config.scale_remap_mu * torch.abs(x) / 6.0) / math.log(
            1.0 + config.scale_remap_mu
        )
    raise ValueError(f"Unsupported scale_remap: {config.scale_remap}")


def _undo_remap(x: torch.Tensor, config: NVFP4Config) -> torch.Tensor:
    if config.scale_remap == "none":
        return x
    if config.scale_remap == "linear":
        return x / config.scale_factor
    if config.scale_remap in ("power", "auto"):
        return torch.sign(x) * 6.0 * (torch.abs(x) / 6.0).pow(1.0 / config.scale_remap_gamma)
    if config.scale_remap == "mulaw":
        import math

        return torch.sign(x) * (6.0 / config.scale_remap_mu) * (
            torch.exp(torch.abs(x) * math.log(1.0 + config.scale_remap_mu) / 6.0) - 1.0
        )
    raise ValueError(f"Unsupported scale_remap: {config.scale_remap}")


def _cast_scale_precision(scales: torch.Tensor, scale_precision: str) -> torch.Tensor:
    if scale_precision == "fp16":
        return scales.to(torch.float16)
    if scale_precision == "bf16":
        return scales.to(torch.bfloat16)
    if scale_precision == "fp32":
        return scales.to(torch.float32)
    if scale_precision in ("e4m3", "e8m0"):
        # Keep this hardware-oriented mode explicit for later packer work.
        return scales.to(torch.float16)
    raise ValueError(f"Unsupported scale precision: {scale_precision}")

