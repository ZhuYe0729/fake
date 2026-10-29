"""Real FP8 E4M3 W8A8 Linear module using torch._scaled_mm.

This module performs true FP8 matrix multiplication on GPU via
torch._scaled_mm in RowWise mode. Key constraints:
  - B (weight) must be stored in column-major layout: stride=(1, in_features)
  - RowWise requires: scale_a=(M,1), scale_b=(1,N), out_dtype=bfloat16
  - FP8 E4M3 max representable value is 448.0
  - Cannot create FP8 tensors via randn/normal/mul — must cast from float16/bfloat16

torch._scaled_mm internally incorporates scales during matmul and
returns a bfloat16 result directly — no separate dequantize step needed.
"""

import torch
from torch import nn
from functools import partial

FP8_E4M3_MAX = 448.0


@torch.no_grad()
def _quantize_dequantize_per_token_fp8(t):
    """Quantize-dequantize activation to FP8 E4M3 range per-token.

    Used as output_quant for attention layers: simulates FP8 precision loss
    by quantizing to FP8 E4M3 range and dequantizing back.
    Returns a single bfloat16 tensor.
    """
    scales = t.abs().max(dim=-1, keepdim=True)[0] / FP8_E4M3_MAX
    scales.clamp_(min=1e-5)
    result = (t.float() / scales).to(torch.float8_e4m3fn).float() * scales
    return result.to(t.dtype)


@torch.no_grad()
def quantize_weight_per_channel_fp8(w):
    """Quantize weight to FP8 E4M3 with per-channel (per output channel) scales.

    Row-major (out_features, in_features) data is physically identical to
    column-major (in_features, out_features) data. We store the original
    (out, in) row-major layout and use .t() in forward to obtain the
    column-major view that _scaled_mm requires.

    Args:
        w: (out_features, in_features) float tensor

    Returns:
        weight_fp8: (out_features, in_features) float8_e4m3fn tensor, row-major
        weight_scale: (1, out_features) float32 tensor — scale per output channel
    """
    scales = w.abs().max(dim=-1, keepdim=True)[0] / FP8_E4M3_MAX
    scales.clamp_(min=1e-5)
    weight_fp8 = (w / scales).to(torch.float8_e4m3fn)

    weight_scale = scales.view(1, -1).contiguous().float()

    return weight_fp8, weight_scale


@torch.no_grad()
def quantize_activation_per_token_fp8(t):
    """Quantize activation to FP8 E4M3 with per-token (per row) scales.

    Args:
        t: (M, hidden_dim) bfloat16/float16 tensor

    Returns:
        act_fp8: (M, hidden_dim) float8_e4m3fn tensor, row-major contiguous
        act_scale: (M, 1) float32 tensor — scale per token/row
    """
    scales = t.abs().max(dim=-1, keepdim=True)[0] / FP8_E4M3_MAX
    scales.clamp_(min=1e-5)
    act_fp8 = (t.float() / scales).to(torch.float8_e4m3fn).contiguous()
    act_scale = scales.contiguous().float()
    return act_fp8, act_scale


@torch.no_grad()
def quantize_activation_per_tensor_fp8(t):
    """Quantize activation to FP8 E4M3 with per-tensor scale."""
    scale = t.abs().max() / FP8_E4M3_MAX
    scale.clamp_(min=1e-5)
    act_fp8 = (t.float() / scale).to(torch.float8_e4m3fn).contiguous()
    # For RowWise _scaled_mm, scale needs shape (M, 1) even for per-tensor
    act_scale = scale.view(1, 1).expand(t.shape[0], 1).contiguous().float()
    return act_fp8, act_scale


class Fp8RealLinear(nn.Module):
    """W8A8 Linear layer using real FP8 E4M3 matmul via torch._scaled_mm.

    Weight is stored as (out_features, in_features) float8_e4m3fn row-major.
    Row-major (out, in) data is physically identical to column-major (in, out)
    data, so .t() in forward gives the column-major view _scaled_mm requires.
    Activation is dynamically quantized per-token in forward().
    torch._scaled_mm with RowWise mode returns bfloat16 directly.
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

        # Weight stored as FP8 E4M3 in (out_features, in_features) row-major.
        # Row-major (out, in) data = column-major (in, out) data physically.
        # .t() in forward gives (in, out) column-major view for _scaled_mm.
        self.register_buffer(
            "weight",
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.float8_e4m3fn,
                requires_grad=False,
            ),
        )
        # Per-channel weight scale: (1, out_features)
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
            self.act_quant = quantize_activation_per_token_fp8
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = quantize_activation_per_tensor_fp8
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            # Output quantization: simulates FP8 precision loss by quantize/dequantize
            # Returns a single bfloat16 tensor, not (fp8, scale) tuple
            self.output_quant = _quantize_dequantize_per_token_fp8
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
        """Forward pass: quantize activation, FP8 matmul via _scaled_mm, add bias.

        Handles both 2D (M, K) and 3D (batch, seq_len, K) inputs.
        """
        # Handle 3D input: (batch, seq_len, hidden_dim) -> (batch*seq_len, hidden_dim)
        input_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.in_features)

        # Quantize activation
        x_fp8, scale_a = self.act_quant(x)

        # _scaled_mm needs B as (K, N) = (in_features, out_features) column-major.
        # self.weight is (out_features, in_features) row-major.
        # .t() gives (in_features, out_features) with strides (1, in_features)
        # which is column-major for (in, out) — and it's a view, no data copy.
        # Row-major (out, in) data is physically identical to column-major (in, out) data.
        weight_cm = self.weight.t()

        # FP8 matmul: result is bfloat16 directly
        result = torch._scaled_mm(
            x_fp8,
            weight_cm,
            scale_a=scale_a,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
        )

        # Add bias
        if self.bias is not None:
            result = result + self.bias

        # Output quantization (for attention q/k/v projections)
        result = self.output_quant(result)

        # Restore 3D shape if needed
        if len(input_shape) == 3:
            result = result.view(input_shape[0], input_shape[1], self.out_features)

        return result

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = Fp8RealLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            weight_fp8, weight_scale = quantize_weight_per_channel_fp8(
                module.weight.data.float()
            )
            new_module.weight = weight_fp8
            new_module.weight_scale = weight_scale
        elif weight_quant == "per_tensor":
            raise NotImplementedError("per_tensor weight quant not yet for real FP8")
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.bfloat16)
        return new_module

    def __repr__(self):
        return (
            f"Fp8RealLinear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, "
            f"weight_quant={self.weight_quant_name}, "
            f"act_quant={self.act_quant_name}, "
            f"output_quant={self.output_quant_name})"
        )


def quantize_llama_like_real_fp8(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    """Replace all Linear layers in a Llama-like model with Fp8RealLinear."""
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
            m.gate_proj = Fp8RealLinear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = Fp8RealLinear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = Fp8RealLinear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            m.q_proj = Fp8RealLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = Fp8RealLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = Fp8RealLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = Fp8RealLinear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model
