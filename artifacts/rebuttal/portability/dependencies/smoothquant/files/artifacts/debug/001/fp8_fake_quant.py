"""FP8 E4M3 fake quant module for SmoothQuant.

Mirrors the existing INT8 fake_quant.W8A8Linear but quantizes to
FP8 E4M3 range (max=448) instead of INT8 range (max=127).
Computes in bfloat16 after quantize/dequantize — simulates FP8
accuracy without real FP8 hardware matmul.
"""

import torch
from torch import nn
from functools import partial

FP8_E4M3_MAX = 448.0


@torch.no_grad()
def quantize_weight_per_channel_absmax_fp8(w):
    """Quantize weight to FP8 E4M3 range with per-channel absmax scales.

    Simulates: round(w / scale) in FP8 E4M3 range, then multiply back.
    Actual computation stays in float.
    """
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0] / FP8_E4M3_MAX
    scales.clamp_(min=1e-5)
    # Quantize-dequantize in float: simulates FP8 E4M3 rounding
    w_quantized = (w / scales).to(torch.float8_e4m3fn).float() * scales
    # In-place update on original tensor (like INT8 fake_quant does)
    w.copy_(w_quantized)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax_fp8(w):
    """Quantize weight to FP8 E4M3 range with per-tensor absmax scale."""
    scales = w.abs().max() / FP8_E4M3_MAX
    scales.clamp_(min=1e-5)
    w_quantized = (w / scales).to(torch.float8_e4m3fn).float() * scales
    w.copy_(w_quantized)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax_fp8(t):
    """Quantize activation to FP8 E4M3 range with per-token absmax scales."""
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0] / FP8_E4M3_MAX
    scales.clamp_(min=1e-5)
    t_quantized = (t.float() / scales).to(torch.float8_e4m3fn).float() * scales
    t.copy_(t_quantized.to(t.dtype))
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax_fp8(t):
    """Quantize activation to FP8 E4M3 range with per-tensor absmax scale."""
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max() / FP8_E4M3_MAX
    scales.clamp_(min=1e-5)
    t_quantized = (t.float() / scales).to(torch.float8_e4m3fn).float() * scales
    t.copy_(t_quantized.to(t.dtype))
    return t


class W8A8Fp8Linear(nn.Module):
    """W8A8 FP8 E4M3 fake quantization Linear layer.

    Quantizes weights and activations to FP8 E4M3 range but
    computes in float16/bfloat16 — simulates FP8 accuracy.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.bfloat16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.bfloat16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = quantize_activation_per_token_absmax_fp8
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = quantize_activation_per_tensor_absmax_fp8
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        # Quantize activation to FP8 E4M3 range (simulated)
        q_x = self.act_quant(x)
        # Compute in bfloat16 (fake quant — actual FP8 matmul not used)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        # Output quantization (for attention layers)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Fp8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        # Quantize weight in-place to FP8 E4M3 range
        weight_bf16 = module.weight.data.to(torch.bfloat16)
        if weight_quant == "per_channel":
            quantize_weight_per_channel_absmax_fp8(weight_bf16)
        elif weight_quant == "per_tensor":
            quantize_weight_per_tensor_absmax_fp8(weight_bf16)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight = weight_bf16
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.bfloat16)
        return new_module

    def __repr__(self):
        return (
            f"W8A8Fp8Linear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, "
            f"weight_quant={self.weight_quant_name}, "
            f"act_quant={self.act_quant_name}, "
            f"output_quant={self.output_quant_name})"
        )


def quantize_llama_like_fp8(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    """Replace all Linear layers in a Llama-like model with W8A8Fp8Linear (fake FP8)."""
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Fp8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W8A8Fp8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W8A8Fp8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            m.q_proj = W8A8Fp8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Fp8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Fp8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Fp8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model
