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

from .canonical_nvfp4 import (
    NVFP4CanonicalWeight,
    canonical_from_linear,
    dequantize_nvfp4_canonical_weight,
    quantize_nvfp4_canonical_weight,
)


_EXT_NAME = "marlin_nvfp4_ext"
_FP4_MAX = 6.0
_E4M3_MAX = 448.0
_MARLIN_GROUP_SIZE = 16


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _standalone_root() -> Path:
    return _repo_root() / "standalone_marlin_nvfp4"


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
    root = _standalone_root()
    source_dir = root / "csrc"
    build_dir = Path(
        os.environ.get(
            "CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR",
            _repo_root() / "artifacts" / "torch_extensions_marlin_nvfp4",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name=_EXT_NAME,
        sources=[
            str(source_dir / "torch_bindings.cpp"),
            str(source_dir / "quantization" / "marlin" / "marlin.cu"),
            str(source_dir / "quantization" / "marlin" / "gptq_marlin_repack.cu"),
            str(source_dir / "quantization" / "marlin" / "sm80_kernel_float16_fe2m1f_float16.cu"),
            str(source_dir / "quantization" / "marlin" / "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu"),
        ],
        extra_include_paths=[str(source_dir)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_120,code=sm_120",
            "-static-global-template-stub=false",
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=os.environ.get("CUTLASS_WRAPPER_MARLIN_NVFP4_VERBOSE_BUILD", "0") == "1",
    )


def _scalar_type_id(
    exponent: int,
    mantissa: int,
    signed: bool,
    bias: int,
    finite_values_only: bool,
    nan_repr: int,
) -> int:
    value = 0
    offset = 0

    def add(member: int, bit_width: int) -> None:
        nonlocal value
        nonlocal offset
        value |= (int(member) & ((1 << bit_width) - 1)) << offset
        offset += bit_width

    add(exponent, 8)
    add(mantissa, 8)
    add(signed, 1)
    add(bias, 32)
    add(finite_values_only, 1)
    add(nan_repr, 8)
    return value


FLOAT4_E2M1F_ID = _scalar_type_id(
    exponent=2,
    mantissa=1,
    signed=True,
    bias=0,
    finite_values_only=True,
    nan_repr=0,
)


@dataclass(frozen=True)
class MarlinNVFP4Weight:
    packed_weight: torch.Tensor
    weight_scale: torch.Tensor
    global_scale: torch.Tensor
    workspace: torch.Tensor
    in_features: int
    out_features: int
    activation_dtype: torch.dtype
    original_dtype: torch.dtype
    bias: torch.Tensor | None = None


def _check_activation_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("activation_dtype must be torch.float16 or torch.bfloat16")


def _check_2d_cuda_weight(weight: torch.Tensor, name: str) -> None:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be 2D")
    if weight.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be float16, bfloat16, or float32")
    if not weight.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_supported_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M/N/K must be positive, got M={m}, N={n}, K={k}")
    if k % 128 != 0:
        raise ValueError(f"K/in_features must be divisible by 128 for Marlin NVFP4, got K={k}")
    if n % 64 != 0:
        raise ValueError(f"N/out_features must be divisible by 64 for Marlin NVFP4, got N={n}")


def can_use_marlin_nvfp4(m: int, n: int, k: int, *, load_extension: bool = False) -> bool:
    if not (m > 0 and n > 0 and k > 0 and k % 128 == 0 and n % 64 == 0):
        return False
    if load_extension:
        try:
            _load_ext()
        except Exception:
            return False
    return True


def _scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])

    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])

    return scale_perm, scale_perm_single


def _marlin_permute_scales(
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    scale_perm, scale_perm_single = _scale_perms()
    if group_size < size_k and group_size != -1:
        scales = scales.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        scales = scales.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return scales.reshape((-1, size_n)).contiguous()


def _marlin_permute_bias(bias: torch.Tensor) -> torch.Tensor:
    original_shape = bias.shape
    _, scale_perm_single = _scale_perms()
    bias = bias.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return bias.reshape(*original_shape).contiguous()


def _nvfp4_compute_scale_factor(
    marlin_scales: torch.Tensor,
    activation_dtype: torch.dtype,
) -> float:
    if activation_dtype == torch.float16:
        return 1.0

    ws_float = marlin_scales.float() * (2**7)
    nonzero_mask = ws_float > 0
    if nonzero_mask.any():
        max_value = ws_float[nonzero_mask].max()
        if max_value < _E4M3_MAX * (2**7):
            return float((_E4M3_MAX * (2**7) / max_value).log2().floor().exp2().item())
    return 1.0


def _nvfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    activation_dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Marlin NVFP4 requires a PyTorch build with torch.float8_e4m3fn")

    marlin_scales = marlin_scales.to(torch.float16)
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(marlin_scales.size(0), -1)

    scale_factor = _nvfp4_compute_scale_factor(marlin_scales, activation_dtype)
    if scale_factor > 1.0:
        marlin_scales = (marlin_scales.float() * scale_factor).to(torch.float16)

    marlin_scales = marlin_scales * (2**7)
    marlin_scales[marlin_scales < 2] = 0
    marlin_scales = marlin_scales.view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()
    return marlin_scales, scale_factor


def _nvfp4_marlin_process_global_scale(
    global_scale: torch.Tensor,
    activation_dtype: torch.dtype,
) -> torch.Tensor:
    if activation_dtype == torch.float16:
        target_exponent = 5
    elif activation_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise ValueError("activation_dtype must be torch.float16 or torch.bfloat16")

    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale.to(torch.float32) * (2.0 ** (exponent_bias - 7))


def _make_workspace(device: torch.device | str | int, max_blocks_per_sm: int = 1) -> torch.Tensor:
    if isinstance(device, int):
        device = torch.device("cuda", device)
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Marlin workspace must be allocated on a CUDA device")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int32, device=device)


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


def quantize_marlin_nvfp4_weight_bf16(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize row-major Linear weight [N, K] into Marlin NVFP4 inputs."""
    canonical = quantize_nvfp4_canonical_weight(weight.contiguous())
    return canonical.packed_weight, canonical.logical_scale, canonical.global_scale


def canonical_to_marlin_nvfp4_weight(
    canonical: NVFP4CanonicalWeight,
    *,
    activation_dtype: torch.dtype,
) -> MarlinNVFP4Weight:
    return _prepare_marlin_weight(
        canonical.packed_weight,
        canonical.logical_scale,
        canonical.global_scale,
        activation_dtype=activation_dtype,
        bias=canonical.bias,
        original_dtype=canonical.original_dtype,
    )


def _prepare_marlin_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    activation_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    original_dtype: torch.dtype | None = None,
) -> MarlinNVFP4Weight:
    _load_ext()
    _check_activation_dtype(activation_dtype)
    if not weight.is_cuda or not weight_scale.is_cuda or not global_scale.is_cuda:
        raise ValueError("weight, weight_scale, and global_scale must be CUDA tensors")
    if weight.dim() != 2:
        raise ValueError("weight must have shape [N, K / 2]")
    if weight.dtype != torch.uint8:
        raise TypeError("weight must be torch.uint8")
    if weight_scale.dim() != 2:
        raise ValueError("weight_scale must have shape [N, K / 16]")

    out_features = int(weight.size(0))
    in_features = int(weight.size(1)) * 2
    _check_supported_shape(1, out_features, in_features)
    if weight_scale.shape != (out_features, in_features // _MARLIN_GROUP_SIZE):
        raise ValueError(
            "weight_scale must have shape [N, K / 16], got "
            f"{tuple(weight_scale.shape)} for N={out_features}, K={in_features}"
        )

    perm = torch.empty(0, dtype=torch.int32, device=weight.device)
    qweight = weight.contiguous().view(torch.int32).T.contiguous()
    packed_weight = torch.ops.marlin_nvfp4_ext.gptq_marlin_repack(
        qweight,
        perm,
        in_features,
        out_features,
        4,
        False,
    )

    marlin_scales = _marlin_permute_scales(
        weight_scale.T.contiguous().to(activation_dtype),
        size_k=in_features,
        size_n=out_features,
        group_size=_MARLIN_GROUP_SIZE,
    )
    marlin_scales, scale_factor = _nvfp4_marlin_process_scales(marlin_scales, activation_dtype)
    marlin_global_scale = _nvfp4_marlin_process_global_scale(global_scale, activation_dtype) / scale_factor

    marlin_bias = None
    if bias is not None:
        if bias.dim() != 1 or bias.numel() != out_features:
            raise ValueError(f"bias must have shape [{out_features}]")
        if bias.device != weight.device:
            raise ValueError("bias must be on the same device as weight")
        marlin_bias = _marlin_permute_bias(bias.to(device=weight.device, dtype=activation_dtype).contiguous())

    return MarlinNVFP4Weight(
        packed_weight=packed_weight,
        weight_scale=marlin_scales,
        global_scale=marlin_global_scale.contiguous(),
        workspace=_make_workspace(weight.device),
        in_features=in_features,
        out_features=out_features,
        activation_dtype=activation_dtype,
        original_dtype=original_dtype or weight.dtype,
        bias=marlin_bias,
    )


@torch.no_grad()
def pack_marlin_nvfp4_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation_dtype: torch.dtype = torch.bfloat16,
    original_dtype: torch.dtype | None = None,
) -> MarlinNVFP4Weight:
    _check_activation_dtype(activation_dtype)
    return canonical_to_marlin_nvfp4_weight(
        quantize_nvfp4_canonical_weight(
            weight.contiguous(),
            bias,
            original_dtype=original_dtype or weight.dtype,
        ),
        activation_dtype=activation_dtype,
    )


class MarlinNVFP4Linear(nn.Module):
    """Inference-only Marlin NVFP4 W4A16 replacement for torch.nn.Linear."""

    in_features: int
    out_features: int

    def __init__(self, weight: MarlinNVFP4Weight) -> None:
        super().__init__()
        self.in_features = weight.in_features
        self.out_features = weight.out_features
        self.activation_dtype = weight.activation_dtype
        self.original_dtype = weight.original_dtype
        self.register_buffer("packed_weight", weight.packed_weight.contiguous(), persistent=True)
        self.register_buffer("weight_scale", weight.weight_scale.contiguous(), persistent=True)
        self.register_buffer("weight_global_scale", weight.global_scale.contiguous(), persistent=True)
        self.register_buffer("workspace", weight.workspace.contiguous(), persistent=False)
        if weight.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", weight.bias.contiguous(), persistent=True)
        # Pre-allocate dummy c_tmp for fast path (not used for W4A16, but must be non-null)
        self.register_buffer(
            "_c_tmp", torch.empty(1, dtype=torch.float32, device=weight.packed_weight.device),
            persistent=False)
        # Fast path is safe: NVFP4 W4A16 BF16, no bias, no act_order, no zp
        self._use_fast_path = (
            weight.activation_dtype == torch.bfloat16
            and weight.bias is None
        )
        self.eval()
        self.requires_grad_(False)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        device: torch.device | str | None = None,
        activation_dtype: torch.dtype = torch.bfloat16,
    ) -> "MarlinNVFP4Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an instance of torch.nn.Linear")
        _check_activation_dtype(activation_dtype)
        target_device = torch.device(device) if device is not None else linear.weight.device
        if target_device.type != "cuda":
            raise ValueError("MarlinNVFP4Linear.from_linear requires a CUDA target device")
        return cls(
            canonical_to_marlin_nvfp4_weight(
                canonical_from_linear(linear, device=target_device),
                activation_dtype=activation_dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("MarlinNVFP4Linear is inference-only; call eval() before forward")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("MarlinNVFP4Linear does not implement autograd; run under torch.no_grad()")
        if x.device.type != "cuda":
            raise ValueError("x must be a CUDA tensor")
        if x.device != self.packed_weight.device:
            raise ValueError("x must be on the same CUDA device as the Marlin NVFP4 weight")
        if x.size(-1) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {x.size(-1)}")

        batch_shape = x.shape[:-1]
        with torch.no_grad():
            _load_ext()

            if self._use_fast_path:
                # Fused C++ path: x.to+reshape+contiguous+kernel+reshape
                # all in a single torch.ops call, eliminating per-op
                # Python→C++ round-trips.
                return torch.ops.marlin_nvfp4_ext.marlin_linear_forward(
                    x,
                    self.packed_weight,
                    self.weight_scale,
                    self.weight_global_scale,
                    self.workspace,
                    self._c_tmp,
                    self.out_features,
                    self.in_features,
                )

            x_flat = x.to(dtype=self.activation_dtype).reshape(-1, self.in_features).contiguous()
            m = int(x_flat.size(0))
            _check_supported_shape(m, self.out_features, self.in_features)
            if not can_use_marlin_nvfp4(m, self.out_features, self.in_features):
                raise RuntimeError(
                    f"Marlin NVFP4 GEMM cannot implement M={m}, "
                    f"N={self.out_features}, K={self.in_features}"
                )
            out = torch.ops.marlin_nvfp4_ext.marlin_gemm(
                x_flat,
                None,
                self.packed_weight,
                self.bias,
                self.weight_scale,
                None,
                self.weight_global_scale,
                None,
                None,
                None,
                self.workspace,
                FLOAT4_E2M1F_ID,
                m,
                self.out_features,
                self.in_features,
                True,
                False,
                True,
                False,
            )
            return out.reshape(*batch_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, activation_dtype={self.activation_dtype}"
        )


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


def marlin_nvfp4_quantize_dequantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    canonical = quantize_nvfp4_canonical_weight(weight.contiguous())
    return dequantize_nvfp4_canonical_weight(canonical), canonical.logical_scale, canonical.global_scale


def simulate_marlin_nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("simulate_marlin_nvfp4_linear expects CUDA tensors")
    _check_activation_dtype(activation_dtype)
    weight_dequant, _scale, _global_scale = marlin_nvfp4_quantize_dequantize(weight.to(torch.bfloat16).contiguous())
    bias_cast = bias.to(device=x.device, dtype=activation_dtype) if bias is not None else None
    out = F.linear(x.to(activation_dtype), weight_dequant.to(activation_dtype), bias_cast)
    return out.to(activation_dtype)


__all__ = [
    "FLOAT4_E2M1F_ID",
    "MarlinNVFP4Linear",
    "MarlinNVFP4Weight",
    "can_use_marlin_nvfp4",
    "canonical_to_marlin_nvfp4_weight",
    "marlin_nvfp4_quantize_dequantize",
    "pack_marlin_nvfp4_weight",
    "quantize_marlin_nvfp4_weight_bf16",
    "simulate_marlin_nvfp4_linear",
]
