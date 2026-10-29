"""Real W8A8 kernels; candidate-local SmoothQuant, including its runtime cost."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from common import RUN


@triton.jit
def _quantize(X, S, Q, A, K: tl.constexpr, M: tl.constexpr,
              LIMIT: tl.constexpr, INTEGER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    smooth = tl.load(S + col, col < K, 1).to(tl.float32)
    value = value / smooth
    scale = tl.maximum(tl.max(tl.abs(value), 0) / LIMIT, 1.e-12)
    result = tl.minimum(tl.maximum(value / scale, -LIMIT), LIMIT)
    if INTEGER:
        result = tl.extra.cuda.libdevice.nearbyint(result)
    tl.store(Q + row * K + col, result, col < K)
    tl.store(A + row, scale)


@triton.jit
def _dequantize(C, A, B, O, N: tl.constexpr, SIZE: tl.constexpr,
                BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(C + index, index < SIZE, 0).to(tl.float32)
    sa = tl.load(A + index // N, index < SIZE, 0)
    sb = tl.load(B + index % N, index < SIZE, 0)
    tl.store(O + index, value * sa * sb, index < SIZE)


def activation(x, smooth, integer):
    m, k = x.shape
    padded = max(m, 32) if integer else m
    dtype = torch.int8 if integer else torch.float8_e4m3fn
    q = torch.empty((padded, k), device=x.device, dtype=dtype)
    scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    # Rows beyond m are discarded, and never enter the result or scales.
    _quantize[(m,)](x, smooth, q, scale, k, m, 127. if integer else 448.,
                    integer, triton.next_power_of_2(k))
    return q, scale


def forward(x, weight, scale, smooth, method):
    shape = x.shape[:-1]
    flat = x.reshape(-1, weight.shape[1]).contiguous()
    integer = method == 'sq_int8'
    quantized, act_scale = activation(flat, smooth, integer)
    if integer:
        accumulator = torch._int_mm(quantized, weight.T)
        out = torch.empty((len(flat), len(weight)), device=x.device, dtype=x.dtype)
        _dequantize[(triton.cdiv(out.numel(), 1024),)](
            accumulator, act_scale, scale, out, len(weight), out.numel(), 1024)
    else:
        from vllm import _custom_ops as ops
        out = ops.cutlass_scaled_mm(quantized, weight.T, act_scale,
                                    scale.T.contiguous(), out_dtype=x.dtype)
    return out.reshape(*shape, weight.shape[0])


def smooth_scale(name, weight, act_scales):
    # Match official Llama-like SmoothQuant: RMSNorm -> QKV and gate/up only.
    if not any(name.endswith(suffix) for suffix in ['qkv_proj', 'gate_up_proj']):
        return torch.ones(weight.shape[1], device=weight.device, dtype=torch.bfloat16)
    maxima = act_scales[name].to(device=weight.device, dtype=weight.dtype)
    weight_max = weight.abs().amax(0).clamp_min(1.e-5)
    return (maxima.pow(.5) / weight_max.pow(.5)).clamp_min(1.e-5).to(weight.dtype)


@torch.no_grad()
def pack(method, name, weight, act_scales=None):
    if method == 'fp8':
        smooth = torch.ones(weight.shape[1], dtype=weight.dtype, device=weight.device)
    else:
        if act_scales is None:
            act_scales = torch.load(RUN/'data/activation_scales.pt', weights_only=True)
        smooth = smooth_scale(name, weight, act_scales)
    transformed = (weight * smooth[None]).to(torch.bfloat16)
    limit = 127. if method == 'sq_int8' else 448.
    scale = transformed.float().abs().amax(1, keepdim=True).clamp_min(1.e-12) / limit
    q = transformed.float() / scale
    if method == 'sq_int8':
        q = q.round().clamp(-127, 127).to(torch.int8)
    else:
        q = q.clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.contiguous(), smooth.contiguous()


class EightBitLinear(torch.nn.Module):
    def __init__(self, method, name, weight, act_scales=None):
        super().__init__()
        self.method = method
        for key, tensor in zip(['weight','scale','smooth'], pack(method,name,weight,act_scales), strict=True):
            self.register_buffer(key, tensor)

    def forward(self, x):
        return forward(x,self.weight,self.scale,self.smooth,self.method)
