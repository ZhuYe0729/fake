from __future__ import annotations

import torch
import torch.nn.functional as F


NVFP4_BLOCK_SIZE = 16
NVFP4_GLOBAL_SCALE_NUMERATOR = 448.0 * 6.0


def nvfp4_global_scale(x: torch.Tensor) -> torch.Tensor:
    clean = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = clean.abs().max().clamp_min(1.0e-12)
    return torch.tensor([NVFP4_GLOBAL_SCALE_NUMERATOR], device=x.device, dtype=torch.float32) / max_abs


def nvfp4_quantize_dequantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("x must be 2D")
    if x.size(1) % NVFP4_BLOCK_SIZE != 0:
        raise ValueError("x last dimension must be divisible by 16")

    x_clean = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    global_scale = nvfp4_global_scale(x_clean)
    rows, cols = x_clean.shape
    groups = cols // NVFP4_BLOCK_SIZE
    grouped = x_clean.reshape(rows, groups, NVFP4_BLOCK_SIZE)
    block_max = grouped.abs().amax(dim=-1)
    raw_scale = torch.where(
        block_max > 0,
        global_scale * block_max / 6.0,
        torch.ones_like(block_max),
    ).clamp(1.0e-8, 448.0)
    block_scale = _quantize_ue4m3(raw_scale)
    scaled = grouped * global_scale / block_scale.unsqueeze(-1)
    fp4_values = _quantize_e2m1_values(scaled.clamp(-6.0, 6.0))
    dequant = fp4_values * block_scale.unsqueeze(-1) / global_scale
    return dequant.reshape_as(x_clean), block_scale, global_scale


def simulate_nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    x_dequant, _, _ = nvfp4_quantize_dequantize(x.to(torch.bfloat16))
    weight_dequant, _, _ = nvfp4_quantize_dequantize(weight.to(torch.bfloat16))
    out = F.linear(x_dequant.float(), weight_dequant.float(), bias.float() if bias is not None else None)
    return out.to(out_dtype)


def _quantize_ue4m3(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "float8_e4m3fn"):
        return x.to(torch.float8_e4m3fn).to(torch.float32)
    return _quantize_e4m3_positive_fallback(x)


def _quantize_e4m3_positive_fallback(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 448.0)
    zero = torch.zeros_like(x)
    safe = x.clamp_min(2.0**-9)
    exponent = torch.floor(torch.log2(safe)).clamp(-6, 8)
    step = torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), exponent - 3)
    rounded = torch.round(safe / step) * step
    return torch.where(x == 0, zero, rounded.clamp(0.0, 448.0))


def _quantize_e2m1_values(x: torch.Tensor) -> torch.Tensor:
    abs_x = x.abs()
    values = torch.empty_like(abs_x)
    values = torch.where(abs_x <= 0.25, torch.zeros_like(values), values)
    values = torch.where((abs_x > 0.25) & (abs_x < 0.75), torch.full_like(values, 0.5), values)
    values = torch.where((abs_x >= 0.75) & (abs_x <= 1.25), torch.full_like(values, 1.0), values)
    values = torch.where((abs_x > 1.25) & (abs_x < 1.75), torch.full_like(values, 1.5), values)
    values = torch.where((abs_x >= 1.75) & (abs_x <= 2.5), torch.full_like(values, 2.0), values)
    values = torch.where((abs_x > 2.5) & (abs_x < 3.5), torch.full_like(values, 3.0), values)
    values = torch.where((abs_x >= 3.5) & (abs_x <= 5.0), torch.full_like(values, 4.0), values)
    values = torch.where(abs_x > 5.0, torch.full_like(values, 6.0), values)
    return torch.copysign(values, x)
