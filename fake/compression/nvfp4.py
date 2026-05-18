from __future__ import annotations

from dataclasses import dataclass

import torch


FP4_E2M1_MAX = 6.0


@dataclass(frozen=True)
class NVFP4Config:
    group_size: int = 16
    scale_precision: str = "fp16"
    scale_rule: str = "static_6"
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
    dequant, scales, scale_rule_stats = _fake_quantize_grouped(grouped, config)
    dequant = dequant.reshape_as(x).to(original_dtype)
    scale_tensor = _cast_scale_precision(scales.squeeze(-1), config.scale_precision)
    stats = {
        "status": "ok",
        "group_size": config.group_size,
        "scale_precision": config.scale_precision,
        "scale_rule": config.scale_rule,
        "scale_remap": config.scale_remap,
        "scale_factor": config.scale_factor,
        "num_groups": int(scale_tensor.numel()),
        "scale_min": float(scales.min().item()),
        "scale_max": float(scales.max().item()),
        **scale_rule_stats,
    }
    return NVFP4Result(dequant, scale_tensor.cpu(), stats)


def fake_quantize_nvfp4_activation(x: torch.Tensor, config: NVFP4Config) -> torch.Tensor:
    if config.group_size not in (16, 32):
        raise ValueError(f"Unsupported activation NVFP4 group_size: {config.group_size}")
    if x.shape[-1] % config.group_size != 0:
        raise ValueError(
            f"Activation last dimension must be divisible by group_size: columns={x.shape[-1]} "
            f"group_size={config.group_size}"
        )
    original_dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], -1, config.group_size)
    dequant, _scales, _stats = _fake_quantize_grouped(grouped, config)
    return dequant.reshape_as(x_float).to(original_dtype)


def _fake_quantize_grouped(grouped: torch.Tensor, config: NVFP4Config) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    if config.scale_rule == "static_6":
        scales = _candidate_scales(grouped, FP4_E2M1_MAX)
        dequant = _fake_quantize_with_scales(grouped, scales, config)
        return dequant, scales, {"scale_denominator_6_groups": int(scales.numel()), "scale_denominator_4_groups": 0}
    if config.scale_rule == "four_over_six_mse":
        scales_6 = _candidate_scales(grouped, 6.0)
        dequant_6 = _fake_quantize_with_scales(grouped, scales_6, config)
        mse_6 = (dequant_6 - grouped).pow(2).mean(dim=-1, keepdim=True)

        scales_4 = _candidate_scales(grouped, 4.0)
        dequant_4 = _fake_quantize_with_scales(grouped, scales_4, config)
        mse_4 = (dequant_4 - grouped).pow(2).mean(dim=-1, keepdim=True)

        use_4 = mse_4 < mse_6
        dequant = torch.where(use_4, dequant_4, dequant_6)
        scales = torch.where(use_4, scales_4, scales_6)
        groups_4 = int(use_4.sum().item())
        groups_total = int(use_4.numel())
        return dequant, scales, {
            "scale_denominator_6_groups": groups_total - groups_4,
            "scale_denominator_4_groups": groups_4,
        }
    raise ValueError(f"Unsupported NVFP4 scale_rule: {config.scale_rule}")


def _candidate_scales(grouped: torch.Tensor, denominator: float) -> torch.Tensor:
    return (grouped.abs().amax(dim=-1, keepdim=True) / denominator).clamp(min=1e-12)


def _fake_quantize_with_scales(grouped: torch.Tensor, scales: torch.Tensor, config: NVFP4Config) -> torch.Tensor:
    normalized = grouped / scales
    normalized = _apply_remap(normalized, config)
    q = cast_to_fp4(normalized)
    return _undo_remap(q, config) * scales


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
