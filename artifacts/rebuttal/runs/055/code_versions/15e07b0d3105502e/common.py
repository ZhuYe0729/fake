from __future__ import annotations

import json
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
MODEL = Path(os.environ['DASO_MODEL_PATH'])
VLLM = Path(os.environ['DASO_VLLM_ROOT'])
WRAPPER = REPO / 'fake/kernels/cutlass/cutlass_wrapper'
sys.path[:0] = [str(WRAPPER), str(VLLM / 'vllm'), str(REPO / 'daso/src')]


def install_sparse_cuda_compatibility() -> None:
    # CUDA 12.8's compiler requires an explicit atomic scope in this CUTLASS
    # instantiation. The isolated header supplies the documented SYSTEM default.
    import cutlass_wrapper.sparse_nvfp4 as sparse
    original_load = sparse.load

    def load_with_header(*args, **kwargs):
        binary = RUN / 'extensions/sparse_nvfp4' / (kwargs['name'] + '.so')
        if binary.exists():
            spec = importlib.util.spec_from_file_location(kwargs['name'], binary)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        kwargs['extra_include_paths'] = [str(RUN / 'code/compat/include'),
                                        *kwargs.get('extra_include_paths', [])]
        kwargs.setdefault('build_directory', str(RUN / 'extensions/sparse_nvfp4'))
        Path(kwargs['build_directory']).mkdir(parents=True, exist_ok=True)
        return original_load(*args, **kwargs)

    sparse.load = load_with_header


install_sparse_cuda_compatibility()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    quantized = (weight.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


def int8_forward(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                 *, backend: str = 'cutlass') -> torch.Tensor:
    if backend == 'int_mm':
        from eightbit import forward
        key = (x.device, x.dtype, weight.shape[1])
        if key not in _INT8_IDENTITIES:
            _INT8_IDENTITIES[key] = torch.ones(weight.shape[1], device=x.device, dtype=x.dtype)
        return forward(x, weight, scale, _INT8_IDENTITIES[key], 'sq_int8')
    shape = x.shape[:-1]
    flat = x.reshape(-1, weight.shape[1]).contiguous()
    activation, activation_scale = quantize_int8(flat)
    if backend == 'cutlass':
        from vllm import _custom_ops as ops
        out = ops.cutlass_scaled_mm(activation, weight.T, activation_scale,
                                    scale.T.contiguous(), out_dtype=x.dtype)
    else:
        tokens = activation.shape[0]
        if tokens < 32:
            activation = torch.nn.functional.pad(activation, (0, 0, 0, 32 - tokens))
        integer = torch._int_mm(activation, weight.T)[:tokens]
        out = (integer.float()
               * activation_scale * scale.T).to(x.dtype)
    return out.reshape(*shape, weight.shape[0])


_INT8_IDENTITIES = {}


class Int8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, *, backend: str = 'cutlass'):
        super().__init__()
        quantized, scale = quantize_int8(weight)
        self.register_buffer('weight', quantized)
        self.register_buffer('scale', scale)
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_forward(x, self.weight, self.scale, backend=self.backend)


class PaddedLinear(nn.Module):
    def __init__(self, module: nn.Module, multiple: int):
        super().__init__()
        self.module = module
        self.multiple = multiple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        padding = (-flat.shape[0]) % self.multiple
        if padding:
            flat = torch.nn.functional.pad(flat, (0, 0, 0, padding))
        output = self.module(flat)[:x.numel() // x.shape[-1]]
        return output.reshape(*shape, output.shape[-1])


def compressed_module(method: str, weight: torch.Tensor,
                      *, int8_backend: str = 'cutlass') -> nn.Module:
    from cutlass_wrapper import (NVFP4Linear, SparseBF16Linear, SparseNVFP4Linear,
        MarlinNVFP4Linear, W4A4Int4Linear, quantize_weight_bf16,
        pack_sparse_bf16_weight, quantize_sparse_weight_bf16,
        pack_marlin_nvfp4_weight, quantize_w4a4_int4_weight)
    if method == 'dense_bf16':
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                          device=weight.device, dtype=weight.dtype)
        layer.weight = nn.Parameter(weight, requires_grad=False)
        return layer
    if method == 'int8':
        return Int8Linear(weight, backend=int8_backend)
    if method == 'int4':
        return W4A4Int4Linear(quantize_w4a4_int4_weight(weight))
    if method == 'marlin_int4':
        from int4_marlin import MarlinInt4
        return MarlinInt4(weight)
    if method == 'dense_nvfp4':
        return NVFP4Linear(quantize_weight_bf16(weight))
    if method == 'sparse_bf16':
        return PaddedLinear(SparseBF16Linear(pack_sparse_bf16_weight(weight, prune=False)), 8)
    if method == 'sparse_nvfp4':
        return PaddedLinear(SparseNVFP4Linear(quantize_sparse_weight_bf16(weight, prune=False)), 32)
    if method == 'w4a16_ours':
        return MarlinNVFP4Linear(pack_marlin_nvfp4_weight(weight))
    raise ValueError(method)
