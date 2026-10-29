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


_EXT_NAME = "_cutlass_wrapper_sparse_nvfp4_sm120"
_FP4_MAX = 6.0
_E4M3_MAX = 448.0
_SPARSE_NVFP4_GROUP_SIZE = 32


def _repo_root() -> Path:
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
    # Remove PyTorch's default arch flags so that only the explicit
    # -gencode=arch=compute_120a,code=sm_120a in extra_cuda_cflags is used.
    # sm_120 does not support block-scaled MMA instructions (with or without .sp modifier).
    cpp_extension.COMMON_NVCC_FLAGS[:] = [
        f for f in cpp_extension.COMMON_NVCC_FLAGS
        if not f.startswith("-gencode") and not f.startswith("-arch=")
    ]


@lru_cache(maxsize=1)
def _load_ext() -> Any:
    _remove_unwanted_pytorch_nvcc_flags()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    root = _repo_root()
    source_dir = root / "cutlass_wrapper" / "csrc"
    build_dir = Path(
        os.environ.get(
            "CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR",
            root / "artifacts" / "torch_extensions",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name=_EXT_NAME,
        sources=[
            str(source_dir / "sparse_nvfp4_bindings.cpp"),
            str(source_dir / "sparse_nvfp4_sm120.cu"),
        ],
        extra_include_paths=[
            str(root / "cutlass" / "include"),
            str(root / "cutlass" / "tools" / "util" / "include"),
            str(root / "cutlass" / "examples" / "common"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=os.environ.get("CUTLASS_WRAPPER_SPARSE_NVFP4_VERBOSE_BUILD", "0") == "1",
    )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _check_2d_cuda_bf16(x: torch.Tensor, name: str) -> None:
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D")
    if x.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be torch.bfloat16")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_sparse_supported_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M/N/K must be positive, got M={m}, N={n}, K={k}")
    if k % 64 != 0:
        raise ValueError(f"K/in_features must be divisible by 64 for SM120 sparse NVFP4, got K={k}")
    if n % 32 != 0:
        raise ValueError(f"N/tokens must be divisible by 32 for SM120 sparse NVFP4 v1, got N={n}")


def can_use_cutlass_sparse_nvfp4(m: int, n: int, k: int, *, load_extension: bool = False) -> bool:
    del load_extension
    return m > 0 and n > 0 and k > 0 and k % 64 == 0 and n % 32 == 0


def magnitude_prune_pairwise_4to8_bf16(x: torch.Tensor) -> torch.Tensor:
    """Keep the two largest adjacent pairs in every group of eight values."""
    if x.dtype != torch.bfloat16:
        raise TypeError("x must be torch.bfloat16")
    if x.numel() % 8 != 0:
        raise ValueError("x.numel() must be divisible by 8 for pairwise 4:8 pruning")

    groups = x.reshape(-1, 4, 2)
    pair_scores = groups.abs().sum(dim=-1)
    _, pair_indices = pair_scores.sort(dim=1)
    pair_mask = torch.ones_like(pair_scores)
    pair_mask.scatter_(1, pair_indices[:, :2], 0)
    return (groups * pair_mask.unsqueeze(-1)).reshape_as(x)


def is_pairwise_4to8_sparse(x: torch.Tensor) -> bool:
    if x.numel() % 8 != 0:
        return False
    groups = x.reshape(-1, 4, 2)
    pair_active = groups.abs().sum(dim=-1) != 0
    return bool((pair_active.sum(dim=1) <= 2).all().item())


def require_pairwise_4to8_sparse(x: torch.Tensor, name: str = "weight") -> None:
    if not is_pairwise_4to8_sparse(x):
        raise ValueError(
            f"{name} must already satisfy pairwise 4:8 structured sparsity; "
            "pass prune=True to SparseNVFP4Linear.from_linear to prune during conversion"
        )


def sparse_nvfp4_global_scale(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("x must be 2D")
    if x.device.type != "cuda":
        raise ValueError("x must be a CUDA tensor")
    clean = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = clean.abs().amax().clamp_min(1.0e-12)
    return torch.tensor([_E4M3_MAX * _FP4_MAX], device=x.device, dtype=torch.float32) / max_abs


def _to_blocked_scale(logical_sf: torch.Tensor) -> torch.Tensor:
    if logical_sf.dim() != 2:
        raise ValueError("logical_sf must be 2D")
    rows, cols = int(logical_sf.size(0)), int(logical_sf.size(1))
    padded_rows = _ceil_div(rows, 128) * 128
    padded_cols = _ceil_div(cols, 4) * 4
    padded = logical_sf
    if padded_rows != rows or padded_cols != cols:
        padded = F.pad(padded, (0, padded_cols - cols, 0, padded_rows - rows))

    row_blocks = padded_rows // 128
    col_blocks = padded_cols // 4
    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    block_tiles = blocks.reshape(row_blocks * col_blocks, 4, 32, 4).transpose(1, 2)
    return block_tiles.reshape(row_blocks * col_blocks, 32, 16).flatten().contiguous()


def _quantize_to_e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=values.device,
        dtype=values.dtype,
    )
    codes = torch.bucketize(values.abs(), thresholds, right=False).to(torch.uint8)
    return (codes + torch.signbit(values).to(torch.uint8) * 8).to(torch.uint8)


def _pack_nibbles(e2m1_codes: torch.Tensor) -> torch.Tensor:
    lo = e2m1_codes[:, 0::2]
    hi = e2m1_codes[:, 1::2]
    return ((hi << 4) | lo).contiguous()


def quantize_sparse_nvfp4_bf16(
    x: torch.Tensor,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a row-major BF16 matrix for the SM120 sparse NVFP4 path.

    The sparse GeForce kernel uses NVFP4/UE4M3 input scale groups of 32 for
    both sparse A and dense B operands.
    """
    _check_2d_cuda_bf16(x, "x")
    rows, k = int(x.size(0)), int(x.size(1))
    if k % _SPARSE_NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"K must be divisible by 32 for sparse NVFP4 quantization, got K={k}")

    if global_scale is None:
        global_scale = sparse_nvfp4_global_scale(x)
    global_scale = global_scale.to(device=x.device, dtype=torch.float32).reshape(1, 1)

    group_count = k // _SPARSE_NVFP4_GROUP_SIZE
    values = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    groups = values.reshape(rows, group_count, _SPARSE_NVFP4_GROUP_SIZE)
    abs_max = groups.abs().amax(dim=-1)
    raw_scale = torch.where(
        abs_max > 0,
        global_scale * abs_max / _FP4_MAX,
        torch.ones_like(abs_max),
    ).clamp(1.0e-8, _E4M3_MAX)

    logical_sf = raw_scale.to(torch.float8_e4m3fn)
    quantized_scale = logical_sf.to(torch.float32)
    normalized = groups * global_scale / quantized_scale.unsqueeze(-1)
    codes = _quantize_to_e2m1_codes(normalized.clamp(-_FP4_MAX, _FP4_MAX)).reshape(rows, k)
    packed = _pack_nibbles(codes)
    return packed, _to_blocked_scale(logical_sf), global_scale.reshape(1).contiguous()


def pack_sparse_nvfp4_a_from_dense(
    packed_a: torch.Tensor,
    *,
    apply_random_2to4_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if packed_a.dim() != 2:
        raise ValueError("packed_a must be 2D [M, K/2]")
    if packed_a.dtype != torch.uint8:
        raise TypeError("packed_a must be torch.uint8")
    return _load_ext().sparse_nvf4_pack_a_from_dense(packed_a.contiguous(), bool(apply_random_2to4_mask))


def pack_sparse_nvfp4_b(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack row-major BF16 activations as sparse SM120 NVFP4 B + SFB."""
    _check_2d_cuda_bf16(x, "x")
    rows, k = int(x.size(0)), int(x.size(1))
    if k % _SPARSE_NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"K must be divisible by 32 for sparse NVFP4 B packing, got K={k}")
    if rows % 32 != 0:
        raise ValueError(f"N/tokens must be divisible by 32 for sparse NVFP4 B packing, got {rows}")
    return _load_ext().sparse_nvf4_pack_b(x)


def sparse_nvfp4_gemm_bf16(
    m: int,
    sparse_a: torch.Tensor,
    metadata: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    if not torch.is_tensor(alpha):
        alpha = torch.tensor([float(alpha)], device=b_packed.device, dtype=torch.float32)
    else:
        alpha = alpha.to(device=b_packed.device, dtype=torch.float32).reshape(1).contiguous()
    n = int(b_packed.size(0))
    k = int(b_packed.size(1)) * 2
    _check_sparse_supported_shape(int(m), n, k)
    return _load_ext().sparse_nvf4_gemm_from_sparse_packed(
        int(m),
        sparse_a.contiguous(),
        metadata.contiguous(),
        b_packed.contiguous(),
        sfa.contiguous(),
        sfb.contiguous(),
        alpha,
    )


@dataclass(frozen=True)
class SparseNVFP4Weight:
    sparse_weight: torch.Tensor
    metadata: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    in_features: int
    out_features: int
    original_dtype: torch.dtype
    bias: torch.Tensor | None = None


@torch.no_grad()
def quantize_sparse_weight_bf16(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    original_dtype: torch.dtype | None = None,
    prune: bool = False,
) -> SparseNVFP4Weight:
    _check_2d_cuda_bf16(weight, "weight")
    out_features, in_features = int(weight.size(0)), int(weight.size(1))
    _check_sparse_supported_shape(out_features, 32, in_features)

    sparse_weight_bf16 = weight.contiguous()
    if prune:
        sparse_weight_bf16 = magnitude_prune_pairwise_4to8_bf16(sparse_weight_bf16).contiguous()
    else:
        require_pairwise_4to8_sparse(sparse_weight_bf16, "weight")

    packed_weight, scale, global_scale = quantize_sparse_nvfp4_bf16(sparse_weight_bf16)
    sparse_weight, metadata = pack_sparse_nvfp4_a_from_dense(packed_weight)
    quantized_bias = None
    if bias is not None:
        if bias.dim() != 1 or bias.numel() != out_features:
            raise ValueError("bias must have shape [out_features]")
        if bias.device != weight.device:
            raise ValueError("bias must be on the same device as weight")
        quantized_bias = bias.detach().to(device=weight.device, dtype=torch.bfloat16).contiguous()

    return SparseNVFP4Weight(
        sparse_weight=sparse_weight,
        metadata=metadata,
        scale=scale,
        global_scale=global_scale,
        in_features=in_features,
        out_features=out_features,
        original_dtype=original_dtype or weight.dtype,
        bias=quantized_bias,
    )


class SparseNVFP4Linear(nn.Module):
    """Inference-only structured sparse NVFP4 replacement for torch.nn.Linear."""

    in_features: int
    out_features: int

    def __init__(self, weight: SparseNVFP4Weight) -> None:
        super().__init__()
        self.in_features = weight.in_features
        self.out_features = weight.out_features
        self.original_dtype = weight.original_dtype
        self.register_buffer("sparse_weight", weight.sparse_weight.contiguous(), persistent=True)
        self.register_buffer("metadata", weight.metadata.contiguous(), persistent=True)
        self.register_buffer("weight_scale", weight.scale.contiguous(), persistent=True)
        self.register_buffer("weight_global_scale", weight.global_scale.contiguous(), persistent=True)
        if weight.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", weight.bias.contiguous(), persistent=True)
        self.eval()
        self.requires_grad_(False)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        device: torch.device | str | None = None,
        prune: bool = False,
    ) -> "SparseNVFP4Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an instance of torch.nn.Linear")
        target_device = torch.device(device) if device is not None else linear.weight.device
        if target_device.type != "cuda":
            raise ValueError("SparseNVFP4Linear.from_linear requires a CUDA target device")
        weight = linear.weight.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
        return cls(quantize_sparse_weight_bf16(weight, bias, original_dtype=linear.weight.dtype, prune=prune))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("SparseNVFP4Linear is inference-only; call eval() before forward")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("SparseNVFP4Linear does not implement autograd; run under torch.no_grad()")
        if x.device.type != "cuda":
            raise ValueError("x must be a CUDA tensor")
        if x.device != self.sparse_weight.device:
            raise ValueError("x must be on the same CUDA device as the SparseNVFP4 weight")
        if x.size(-1) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {x.size(-1)}")

        batch_shape = x.shape[:-1]
        with torch.no_grad():
            x_flat = x.to(dtype=torch.bfloat16).reshape(-1, self.in_features).contiguous()
            tokens = int(x_flat.size(0))
            _check_sparse_supported_shape(self.out_features, tokens, self.in_features)
            if not can_use_cutlass_sparse_nvfp4(self.out_features, tokens, self.in_features):
                raise RuntimeError(
                    f"CUTLASS sparse NVFP4 GEMM cannot implement M={self.out_features}, "
                    f"N={tokens}, K={self.in_features}"
                )

            b_packed, b_scale, b_global_scale = pack_sparse_nvfp4_b(x_flat)
            alpha = torch.reciprocal(self.weight_global_scale * b_global_scale)
            out_storage = sparse_nvfp4_gemm_bf16(
                self.out_features,
                self.sparse_weight,
                self.metadata,
                b_packed,
                self.weight_scale,
                b_scale,
                alpha,
            )
            # The SM120 sparse kernel uses ColumnMajor D.  The returned tensor
            # owns the right storage, but its default PyTorch row-major shape is
            # not the logical Linear output. Reinterpret storage as [tokens, out].
            out = out_storage.as_strided((tokens, self.out_features), (self.out_features, 1)).contiguous()
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*batch_shape, self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def _unpack_nibbles(packed: torch.Tensor, k: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.empty((packed.size(0), packed.size(1) * 2), device=packed.device, dtype=torch.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    return codes[:, :k]


def _e2m1_values_from_codes(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device, dtype=torch.float32)
    sign = torch.where((codes & 0x08) != 0, -1.0, 1.0)
    return table[(codes & 0x07).long()] * sign


def simulate_sparse_nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    prune: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference path for tests: sparse/prune then use the same Python quantizer and dequantize."""
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("simulate_sparse_nvfp4_linear expects CUDA tensors")
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    if prune:
        weight_bf16 = magnitude_prune_pairwise_4to8_bf16(weight_bf16).contiguous()
    else:
        require_pairwise_4to8_sparse(weight_bf16, "weight")
    x_2d = x.to(torch.bfloat16).reshape(-1, weight_bf16.size(1)).contiguous()

    w_dequant, _w_scale, _w_global = sparse_nvfp4_quantize_dequantize(weight_bf16)
    x_dequant, _x_scale, _x_global = sparse_nvfp4_quantize_dequantize(x_2d)
    out = F.linear(x_dequant.float(), w_dequant.float(), bias.float() if bias is not None else None)
    return out.to(out_dtype).reshape(*x.shape[:-1], weight_bf16.size(0))


def sparse_nvfp4_quantize_dequantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, scale, global_scale = quantize_sparse_nvfp4_bf16(x.contiguous())
    codes = _unpack_nibbles(packed, int(x.size(1)))
    values = _e2m1_values_from_codes(codes)
    logical_scale = _logical_scales_from_blocked(scale, int(x.size(0)), int(x.size(1)))
    dequant = values.reshape(x.size(0), -1, _SPARSE_NVFP4_GROUP_SIZE)
    dequant = dequant * logical_scale.to(torch.float32).unsqueeze(-1) / global_scale.reshape(1, 1, 1)
    return dequant.reshape_as(x.float()), logical_scale.to(torch.float32), global_scale


def _logical_scales_from_blocked(blocked: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    cols = k // _SPARSE_NVFP4_GROUP_SIZE
    padded_rows = _ceil_div(rows, 128) * 128
    padded_cols = _ceil_div(cols, 4) * 4
    row_blocks = padded_rows // 128
    col_blocks = padded_cols // 4
    tiles = blocked.reshape(row_blocks * col_blocks, 32, 4, 4).transpose(1, 2)
    blocks = tiles.reshape(row_blocks, col_blocks, 128, 4).permute(0, 2, 1, 3)
    logical = blocks.reshape(padded_rows, padded_cols)
    return logical[:rows, :cols]


__all__ = [
    "SparseNVFP4Linear",
    "SparseNVFP4Weight",
    "can_use_cutlass_sparse_nvfp4",
    "is_pairwise_4to8_sparse",
    "magnitude_prune_pairwise_4to8_bf16",
    "pack_sparse_nvfp4_a_from_dense",
    "pack_sparse_nvfp4_b",
    "quantize_sparse_nvfp4_bf16",
    "quantize_sparse_weight_bf16",
    "require_pairwise_4to8_sparse",
    "simulate_sparse_nvfp4_linear",
    "sparse_nvfp4_gemm_bf16",
    "sparse_nvfp4_global_scale",
    "sparse_nvfp4_quantize_dequantize",
]
