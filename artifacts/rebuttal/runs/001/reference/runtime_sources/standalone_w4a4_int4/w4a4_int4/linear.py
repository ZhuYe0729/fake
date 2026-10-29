from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load


_EXT_NAME = "w4a4_int4_ext"
_QMIN = -8
_QMAX = 7


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _standalone_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _remove_unwanted_pytorch_nvcc_flags() -> None:
    for flag in [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]:
        try:
            cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


@lru_cache(maxsize=1)
def _load_ext() -> Any:
    _remove_unwanted_pytorch_nvcc_flags()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

    root = _repo_root()
    source_dir = _standalone_root() / "csrc"
    build_dir = Path(
        os.environ.get(
            "W4A4_INT4_EXT_BUILD_DIR",
            root / "artifacts" / "torch_extensions_w4a4_int4",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name=_EXT_NAME,
        sources=[
            str(source_dir / "torch_bindings.cpp"),
            str(source_dir / "w4a4_int4_kernel.cu"),
        ],
        extra_include_paths=[
            str(root / "cutlass" / "include"),
            str(root / "cutlass" / "tools" / "util" / "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=os.environ.get("W4A4_INT4_VERBOSE_BUILD", "0") == "1",
    )


def _check_2d_cuda_float(x: torch.Tensor, name: str) -> None:
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D")
    if x.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be float16, bfloat16, or float32")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_supported_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M/N/K must be positive, got M={m}, N={n}, K={k}")
    if k % 32 != 0:
        raise ValueError(f"K/in_features must be divisible by 32 for W4A4 INT4, got K={k}")
    if n % 32 != 0:
        raise ValueError(f"N/out_features must be divisible by 32 for W4A4 INT4, got N={n}")


def can_use_w4a4_int4(m: int, n: int, k: int, *, load_extension: bool = False) -> bool:
    if not (m > 0 and n > 0 and k > 0 and k % 32 == 0 and n % 32 == 0):
        return False
    if load_extension:
        try:
            return bool(_load_ext().can_implement_w4a4_int4(int(m), int(n), int(k)))
        except Exception:
            return False
    return True


def _safe_symmetric_scale(max_abs: torch.Tensor) -> torch.Tensor:
    return torch.clamp(max_abs / float(_QMAX), min=1.0e-8)


def _quantize_symmetric(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x.float() / scale), _QMIN, _QMAX).to(torch.int8)


def _pack_signed_int4_row_major(q: torch.Tensor) -> torch.Tensor:
    if q.dtype != torch.int8:
        raise TypeError("q must be torch.int8")
    if q.dim() != 2:
        raise ValueError("q must be 2D")
    if q.size(1) % 2 != 0:
        raise ValueError("last dimension must be even")
    q_u4 = (q.to(torch.int16) & 0x0F).to(torch.uint8)
    lo = q_u4[:, 0::2]
    hi = q_u4[:, 1::2] << 4
    return (lo | hi).contiguous()


def _unpack_signed_int4_row_major(packed: torch.Tensor, cols: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty((packed.size(0), packed.size(1) * 2), device=packed.device, dtype=torch.int8)
    out[:, 0::2] = lo.to(torch.int8)
    out[:, 1::2] = hi.to(torch.int8)
    out = torch.where(out >= 8, out - 16, out)
    return out[:, :cols].contiguous()


def pack_w4a4_int4_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize and pack activation [M, K] as signed int4."""
    _check_2d_cuda_float(x, "x")
    if x.size(1) % 32 != 0:
        raise ValueError(f"K must be divisible by 32 for W4A4 INT4, got {x.size(1)}")
    return _load_ext().pack_activation(x)


@dataclass(frozen=True)
class W4A4Int4Weight:
    packed_weight: torch.Tensor
    weight_scale: torch.Tensor
    in_features: int
    out_features: int
    original_dtype: torch.dtype
    bias: torch.Tensor | None = None


@torch.no_grad()
def quantize_w4a4_int4_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    original_dtype: torch.dtype | None = None,
) -> W4A4Int4Weight:
    """Quantize dense Linear weight [out_features, in_features] into signed int4 storage."""
    _check_2d_cuda_float(weight, "weight")
    n = int(weight.size(0))
    k = int(weight.size(1))
    _check_supported_shape(1, n, k)
    packed, scale = _load_ext().pack_weight(weight)
    packed = packed.contiguous()
    scale = scale.contiguous()
    packed_shape = (n, k // 2)
    if tuple(packed.shape) != packed_shape:
        raise RuntimeError(f"packed weight shape mismatch: expected {packed_shape}, got {tuple(packed.shape)}")

    stored_bias = None
    if bias is not None:
        if bias.dim() != 1 or bias.numel() != n:
            raise ValueError(f"bias must have shape ({n},)")
        stored_bias = bias.to(device=weight.device, dtype=torch.bfloat16).contiguous()

    return W4A4Int4Weight(
        packed_weight=packed,
        weight_scale=scale,
        in_features=k,
        out_features=n,
        original_dtype=original_dtype or weight.dtype,
        bias=stored_bias,
    )


def w4a4_int4_gemm(
    packed_activation: torch.Tensor,
    activation_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    _check_supported_shape(int(m), int(n), int(k))
    return _load_ext().w4a4_int4_gemm(
        packed_activation,
        activation_scale,
        packed_weight,
        weight_scale,
        bias,
        int(m),
        int(n),
        int(k),
    )


def simulate_w4a4_int4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation for the first W4A4 INT4 quantization semantics."""
    if x.size(-1) != weight.size(1):
        raise ValueError(f"expected input last dim {weight.size(1)}, got {x.size(-1)}")
    x_flat = x.reshape(-1, weight.size(1)).float()
    w = weight.float()

    a_scale = _safe_symmetric_scale(torch.nan_to_num(x_flat, 0.0, 0.0, 0.0).abs().amax(dim=1, keepdim=True))
    w_scale = _safe_symmetric_scale(torch.nan_to_num(w, 0.0, 0.0, 0.0).abs().amax(dim=1, keepdim=True))
    q_a = _quantize_symmetric(torch.nan_to_num(x_flat, 0.0, 0.0, 0.0), a_scale)
    q_w = _quantize_symmetric(torch.nan_to_num(w, 0.0, 0.0, 0.0), w_scale)
    out = (q_a.float() @ q_w.float().t()) * a_scale * w_scale.t()
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*x.shape[:-1], weight.size(0)).to(torch.bfloat16)


class W4A4Int4Linear(nn.Module):
    """Inference-only signed-int4 W4A4 replacement for torch.nn.Linear."""

    in_features: int
    out_features: int

    def __init__(self, weight: W4A4Int4Weight) -> None:
        super().__init__()
        self.in_features = weight.in_features
        self.out_features = weight.out_features
        self.original_dtype = weight.original_dtype
        self.register_buffer("packed_weight", weight.packed_weight.contiguous(), persistent=True)
        self.register_buffer("weight_scale", weight.weight_scale.contiguous(), persistent=True)
        if weight.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", weight.bias.contiguous(), persistent=True)
        self.eval()
        self.requires_grad_(False)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, *, device: torch.device | str | None = None) -> "W4A4Int4Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an instance of torch.nn.Linear")
        target_device = torch.device(device) if device is not None else linear.weight.device
        if target_device.type != "cuda":
            raise ValueError("W4A4Int4Linear.from_linear requires a CUDA target device")
        weight = linear.weight.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
        return cls(quantize_w4a4_int4_weight(weight, bias, original_dtype=linear.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("W4A4Int4Linear is inference-only; call eval() before forward")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("W4A4Int4Linear does not implement autograd; run under torch.no_grad()")
        if x.device.type != "cuda":
            raise ValueError("x must be a CUDA tensor")
        if x.device != self.packed_weight.device:
            raise ValueError("x must be on the same CUDA device as the W4A4 INT4 weight")
        if x.size(-1) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {x.size(-1)}")

        batch_shape = x.shape[:-1]
        with torch.no_grad():
            x_flat = x.reshape(-1, self.in_features).contiguous()
            m = int(x_flat.size(0))
            _check_supported_shape(m, self.out_features, self.in_features)
            if not can_use_w4a4_int4(m, self.out_features, self.in_features):
                raise RuntimeError(
                    f"W4A4 INT4 GEMM cannot implement M={m}, "
                    f"N={self.out_features}, K={self.in_features}"
                )
            a_packed, a_scale = pack_w4a4_int4_activation(x_flat)
            out = w4a4_int4_gemm(
                a_packed,
                a_scale,
                self.packed_weight,
                self.weight_scale,
                self.bias,
                m,
                self.out_features,
                self.in_features,
            )
            return out.reshape(*batch_shape, self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


__all__ = [
    "W4A4Int4Linear",
    "W4A4Int4Weight",
    "can_use_w4a4_int4",
    "pack_w4a4_int4_activation",
    "quantize_w4a4_int4_weight",
    "simulate_w4a4_int4_linear",
    "w4a4_int4_gemm",
]
