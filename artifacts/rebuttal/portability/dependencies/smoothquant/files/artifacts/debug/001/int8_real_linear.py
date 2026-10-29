"""Real INT8 W8A8 Linear module using torch._int_mm.

This module performs true INT8 matrix multiplication on GPU via
torch._int_mm, which returns int32 results that we dequantize
using per-token activation scales and per-channel weight scales.

torch._int_mm requires M > 16 strictly. For small batches (M < 17),
we pad the input with zero rows to reach M=17 and strip the padding
from the output afterward.
"""

import torch
from torch import nn
from functools import partial


# Minimum M dimension for torch._int_mm
_INT_MM_MIN_M = 17


@torch.no_grad()
def quantize_weight_per_channel_int8(w):
    """Quantize weight to INT8 with per-channel (per output channel) scales.

    Stores weight in transposed format (in_features, out_features) for
    torch._int_mm which expects (M, K) x (K, N).

    Args:
        w: (out_features, in_features) float tensor

    Returns:
        weight_int8: (in_features, out_features) int8 tensor — transposed for _int_mm
        weight_scale: (1, out_features) float tensor — scale per output channel
    """
    scales = w.abs().max(dim=-1, keepdim=True)[0]  # (out_features, 1)
    q_max = 127.0
    scales.clamp_(min=1e-5).div_(q_max)
    weight_int8 = (w / scales).round().clamp(-127, 127).to(torch.int8)
    # Transpose weight for _int_mm: (out, in) -> (in, out)
    weight_int8_t = weight_int8.t().contiguous()
    # Reshape scales: (out_features, 1) -> (1, out_features)
    scales_t = scales.view(1, -1).contiguous()
    return weight_int8_t, scales_t


@torch.no_grad()
def quantize_activation_per_token_int8(t, n_bits=8):
    """Quantize activation to INT8 with per-token (per row) scales.

    Args:
        t: (batch, hidden_dim) float tensor

    Returns:
        act_int8: (batch, hidden_dim) int8 tensor
        act_scale: (batch, 1) float tensor — scale per token/row
    """
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    act_int8 = (t / scales).round().clamp(-q_max, q_max).to(torch.int8)
    return act_int8, scales


@torch.no_grad()
def _quantize_dequantize_per_token_int8(t, n_bits=8):
    """Quantize-dequantize activation to INT8 range per-token.

    Used as output_quant for attention layers: simulates precision loss
    by quantizing to INT8 range and dequantizing back, returns single tensor.
    """
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    # Quantize then dequantize — simulates INT8 rounding error
    result = (t / scales).round().clamp(-q_max, q_max) * scales
    return result.to(t.dtype)


class Int8RealLinear(nn.Module):
    """W8A8 Linear layer using real INT8 matmul via torch._int_mm.

    Weight is stored as int8 with per-channel scales.
    Activation is dynamically quantized per-token in forward().
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

        # Weight stored as int8 in (in_features, out_features) format (transposed)
        # because torch._int_mm expects (M, K) x (K, N) where K=in_features, N=out_features
        self.register_buffer(
            "weight",
            torch.zeros(
                self.in_features,
                self.out_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(
                (1, self.out_features),
                dtype=torch.float32,
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
            self.act_quant = partial(
                quantize_activation_per_token_int8, n_bits=8
            )
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(
                quantize_activation_per_tensor_int8, n_bits=8
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            # Output quantization: simulates precision loss by quantize/dequantize
            # Returns a single float tensor, not (int8, scale) tuple
            self.output_quant = partial(
                _quantize_dequantize_per_token_int8, n_bits=8
            )
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.weight_scale = self.weight_scale.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        """Forward pass: quantize activation, INT8 matmul, dequantize result.

        Handles both 2D (M, K) and 3D (batch, seq_len, K) inputs.
        For torch._int_mm, M must be > 16. If M < 17 rows,
        we pad with zeros to M=17, compute, then strip padding.
        """
        # Handle 3D input: (batch, seq_len, hidden_dim) -> (batch*seq_len, hidden_dim)
        input_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.in_features)

        original_M = x.shape[0]

        # Quantize activation
        x_int8, act_scale = self.act_quant(x)

        # Pad if M < 17
        needs_padding = original_M < _INT_MM_MIN_M
        if needs_padding:
            pad_rows = _INT_MM_MIN_M - original_M
            x_int8 = torch.cat(
                [x_int8, torch.zeros(pad_rows, self.in_features, dtype=torch.int8, device=x_int8.device)],
                dim=0,
            )
            # Pad act_scale too (scale=1 for zero rows avoids division issues)
            act_scale = torch.cat(
                [act_scale, torch.ones(pad_rows, 1, dtype=act_scale.dtype, device=act_scale.device)],
                dim=0,
            )

        # INT8 matmul: (M, K) x (K, N) -> (M, N) int32
        # weight is (in_features, out_features) = (K, N)
        result_int32 = torch._int_mm(x_int8, self.weight)

        # Dequantize: result_float = int32 * act_scale * weight_scale
        # act_scale: (M, 1), weight_scale: (1, out_features) -> broadcast to (M, out_features)
        result_float = result_int32.float() * act_scale * self.weight_scale

        # Strip padding
        if needs_padding:
            result_float = result_float[:original_M]

        # Add bias
        if self.bias is not None:
            result_float = result_float + self.bias

        # Output quantization (for attention layers)
        result_float = self.output_quant(result_float)

        # Restore 3D shape if needed
        if len(input_shape) == 3:
            result_float = result_float.view(input_shape[0], input_shape[1], self.out_features)

        return result_float.to(x.dtype)

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = Int8RealLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            weight_int8, weight_scale = quantize_weight_per_channel_int8(
                module.weight.float()
            )
            new_module.weight = weight_int8
            new_module.weight_scale = weight_scale
        elif weight_quant == "per_tensor":
            raise NotImplementedError("per_tensor weight quant not yet implemented for real INT8")
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.bfloat16)
        return new_module

    def __repr__(self):
        return (
            f"Int8RealLinear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, "
            f"weight_quant={self.weight_quant_name}, "
            f"act_quant={self.act_quant_name}, "
            f"output_quant={self.output_quant_name})"
        )


@torch.no_grad()
def quantize_activation_per_tensor_int8(t, n_bits=8):
    """Quantize activation to INT8 with per-tensor scale."""
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    act_int8 = (t / scales).round().clamp(-q_max, q_max).to(torch.int8)
    return act_int8, scales.view(1, 1)


def quantize_llama_like_real_int8(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    """Replace all Linear layers in a Llama-like model with Int8RealLinear."""
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
            m.gate_proj = Int8RealLinear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = Int8RealLinear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = Int8RealLinear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            m.q_proj = Int8RealLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = Int8RealLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = Int8RealLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = Int8RealLinear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model
