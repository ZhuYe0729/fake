from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


NVFP4_GROUP_SIZE = 16
NVFP4_FP4_MAX = 6.0
NVFP4_E4M3_MAX = 448.0


@dataclass(frozen=True)
class NVFP4CanonicalWeight:
    """Kernel-neutral dense NVFP4 weight storage.

    Dequantization semantics are:
      fp4_value * logical_scale * global_scale
    """

    packed_weight: torch.Tensor
    logical_scale: torch.Tensor
    global_scale: torch.Tensor
    in_features: int
    out_features: int
    original_dtype: torch.dtype
    bias: torch.Tensor | None = None


def _check_2d_cuda_float_weight(weight: torch.Tensor, name: str) -> None:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be 2D")
    if weight.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be float16, bfloat16, or float32")
    if not weight.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_canonical_shape(out_features: int, in_features: int) -> None:
    if out_features <= 0 or in_features <= 0:
        raise ValueError(f"in/out features must be positive, got out={out_features}, in={in_features}")
    if in_features % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"in_features must be divisible by 16 for NVFP4, got {in_features}")


def _quantize_to_e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=values.device,
        dtype=values.dtype,
    )
    codes = torch.bucketize(values.abs(), thresholds, right=False).to(torch.uint8)
    return (codes + torch.signbit(values).to(torch.uint8) * 8).to(torch.uint8)


def _pack_nibbles(e2m1_codes: torch.Tensor) -> torch.Tensor:
    low = e2m1_codes[:, 0::2]
    high = e2m1_codes[:, 1::2]
    return ((high << 4) | low).contiguous()


@torch.no_grad()
def quantize_nvfp4_canonical_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    original_dtype: torch.dtype | None = None,
) -> NVFP4CanonicalWeight:
    """Quantize row-major Linear weight [out_features, in_features] once."""
    _check_2d_cuda_float_weight(weight, "weight")
    out_features, in_features = int(weight.size(0)), int(weight.size(1))
    _check_canonical_shape(out_features, in_features)

    values = torch.nan_to_num(weight.float(), nan=0.0, posinf=0.0, neginf=0.0)
    group_count = in_features // NVFP4_GROUP_SIZE
    groups = values.reshape(out_features, group_count, NVFP4_GROUP_SIZE)
    abs_max = groups.abs().amax(dim=-1)
    raw_scale = abs_max / NVFP4_FP4_MAX
    max_raw_scale = raw_scale.max().clamp_min(1.0e-12)
    global_scale = (max_raw_scale / NVFP4_E4M3_MAX).reshape(1).contiguous()
    scale = (raw_scale / global_scale.reshape(1, 1)).clamp(1.0e-8, NVFP4_E4M3_MAX)
    logical_scale = scale.to(torch.float8_e4m3fn).contiguous()
    quantized_scale = logical_scale.to(torch.float32)
    normalized = groups / (global_scale.reshape(1, 1, 1) * quantized_scale.unsqueeze(-1))
    codes = _quantize_to_e2m1_codes(normalized.clamp(-NVFP4_FP4_MAX, NVFP4_FP4_MAX)).reshape(
        out_features,
        in_features,
    )

    packed_bias = None
    if bias is not None:
        if bias.dim() != 1 or bias.numel() != out_features:
            raise ValueError("bias must have shape [out_features]")
        if bias.device != weight.device:
            raise ValueError("bias must be on the same device as weight")
        packed_bias = bias.detach().to(device=weight.device, dtype=torch.bfloat16).contiguous()

    return NVFP4CanonicalWeight(
        packed_weight=_pack_nibbles(codes),
        logical_scale=logical_scale,
        global_scale=global_scale,
        in_features=in_features,
        out_features=out_features,
        original_dtype=original_dtype or weight.dtype,
        bias=packed_bias,
    )


@torch.no_grad()
def canonical_from_linear(
    linear: nn.Linear,
    *,
    device: torch.device | str | None = None,
) -> NVFP4CanonicalWeight:
    if not isinstance(linear, nn.Linear):
        raise TypeError("linear must be an instance of torch.nn.Linear")
    target_device = torch.device(device) if device is not None else linear.weight.device
    if target_device.type != "cuda":
        raise ValueError("canonical_from_linear requires a CUDA target device")
    weight = linear.weight.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
    bias = None
    if linear.bias is not None:
        bias = linear.bias.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
    return quantize_nvfp4_canonical_weight(weight, bias, original_dtype=linear.weight.dtype)


def _unpack_nibbles(packed: torch.Tensor, k: int) -> torch.Tensor:
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.empty((packed.size(0), packed.size(1) * 2), device=packed.device, dtype=torch.uint8)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    return codes[:, :k]


def _e2m1_values_from_codes(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device, dtype=torch.float32)
    sign = torch.where((codes & 0x08) != 0, -1.0, 1.0)
    return table[(codes & 0x07).long()] * sign


def dequantize_nvfp4_canonical_weight(canonical: NVFP4CanonicalWeight) -> torch.Tensor:
    codes = _unpack_nibbles(canonical.packed_weight, canonical.in_features)
    values = _e2m1_values_from_codes(codes)
    logical_scale = canonical.logical_scale.to(torch.float32).repeat_interleave(NVFP4_GROUP_SIZE, dim=1)
    return (values * logical_scale * canonical.global_scale.reshape(1, 1)).reshape(
        canonical.out_features,
        canonical.in_features,
    )


def simulate_nvfp4_canonical_linear(
    x: torch.Tensor,
    canonical: NVFP4CanonicalWeight,
    *,
    activation_dtype: torch.dtype = torch.bfloat16,
    quantize_activation: bool = False,
) -> torch.Tensor:
    if x.device.type != "cuda":
        raise ValueError("simulate_nvfp4_canonical_linear expects CUDA input")
    weight = dequantize_nvfp4_canonical_weight(canonical)
    bias = canonical.bias.to(device=x.device, dtype=activation_dtype) if canonical.bias is not None else None
    x_cast = x.to(activation_dtype)
    if quantize_activation:
        from .nvfp4_reference import nvfp4_quantize_dequantize

        x_2d = x_cast.reshape(-1, canonical.in_features).contiguous()
        x_dequant, _scale, _global = nvfp4_quantize_dequantize(x_2d)
        out = F.linear(x_dequant.to(activation_dtype), weight.to(activation_dtype), bias)
        return out.reshape(*x.shape[:-1], canonical.out_features).to(activation_dtype)
    return F.linear(x_cast, weight.to(activation_dtype), bias).to(activation_dtype)


class HybridNVFP4Linear(nn.Module):
    """Shared-weight Linear with W4A4 prefill and W4A16 decode paths."""

    in_features: int
    out_features: int

    def __init__(
        self,
        canonical: NVFP4CanonicalWeight,
        *,
        decode_activation_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = canonical.in_features
        self.out_features = canonical.out_features
        self.original_dtype = canonical.original_dtype
        self.decode_activation_dtype = decode_activation_dtype
        self.register_buffer("canonical_packed_weight", canonical.packed_weight.contiguous(), persistent=True)
        self.register_buffer("canonical_logical_scale", canonical.logical_scale.contiguous(), persistent=True)
        self.register_buffer("canonical_global_scale", canonical.global_scale.contiguous(), persistent=True)
        if canonical.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", canonical.bias.contiguous(), persistent=True)
        object.__setattr__(self, "_cutlass_linear", None)
        object.__setattr__(self, "_marlin_linear", None)
        self.eval()
        self.requires_grad_(False)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        device: torch.device | str | None = None,
        decode_activation_dtype: torch.dtype = torch.bfloat16,
    ) -> "HybridNVFP4Linear":
        return cls(
            canonical_from_linear(linear, device=device),
            decode_activation_dtype=decode_activation_dtype,
        )

    def _canonical(self) -> NVFP4CanonicalWeight:
        return NVFP4CanonicalWeight(
            packed_weight=self.canonical_packed_weight,
            logical_scale=self.canonical_logical_scale,
            global_scale=self.canonical_global_scale,
            in_features=self.in_features,
            out_features=self.out_features,
            original_dtype=self.original_dtype,
            bias=self.bias,
        )

    def _get_cutlass_linear(self) -> nn.Module:
        if self._cutlass_linear is None:
            from .nvfp4 import NVFP4Linear, canonical_to_cutlass_nvfp4_weight

            object.__setattr__(
                self,
                "_cutlass_linear",
                NVFP4Linear(canonical_to_cutlass_nvfp4_weight(self._canonical())).eval(),
            )
        return self._cutlass_linear

    def _get_marlin_linear(self) -> nn.Module:
        if self._marlin_linear is None:
            from .marlin_nvfp4 import MarlinNVFP4Linear, canonical_to_marlin_nvfp4_weight

            object.__setattr__(
                self,
                "_marlin_linear",
                MarlinNVFP4Linear(
                    canonical_to_marlin_nvfp4_weight(
                        self._canonical(),
                        activation_dtype=self.decode_activation_dtype,
                    )
                )
                .eval(),
            )
        return self._marlin_linear

    def forward_w4a4(self, x: torch.Tensor) -> torch.Tensor:
        return self._get_cutlass_linear()(x)

    def forward_w4a16(self, x: torch.Tensor) -> torch.Tensor:
        return self._get_marlin_linear()(x)

    def forward(self, x: torch.Tensor, *, decode: bool = False) -> torch.Tensor:
        if decode:
            return self.forward_w4a16(x)
        return self.forward_w4a4(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"decode_activation_dtype={self.decode_activation_dtype}"
        )


__all__ = [
    "HybridNVFP4Linear",
    "NVFP4CanonicalWeight",
    "canonical_from_linear",
    "dequantize_nvfp4_canonical_weight",
    "quantize_nvfp4_canonical_weight",
    "simulate_nvfp4_canonical_linear",
]
