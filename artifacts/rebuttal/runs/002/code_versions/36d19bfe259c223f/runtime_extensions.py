"""Experiment-local additions to the unchanged paper phase runtime."""
from __future__ import annotations

import torch

from common import int8_forward


def install():
    from vllm.model_executor.layers.quantization import phase_hetero_mytest as phase
    from vllm.model_executor.models.qwen2 import Qwen2Model

    # Qwen3 inherits this loader. Its raw-HF mapping would replace "up_proj"
    # inside the already fused "gate_up_proj" of the paper export a second time.
    original_loader = Qwen2Model.load_weights

    def load_weights(model, weights):
        parameters = dict(model.named_parameters())
        loaded = set()

        def unfused_weights():
            for name, value in weights:
                if '.ph_prefill_' in name or '.ph_decode_' in name:
                    parameter = parameters[name]
                    parameter.weight_loader(parameter, value)
                    loaded.add(name)
                else:
                    yield name, value

        loaded.update(original_loader(model, unfused_weights()))
        return loaded

    Qwen2Model.load_weights = load_weights

    class IntegerAdapter(phase.LinearAdapter):
        names = ('packed_weight', 'weight_scale')

        def __init__(self, method):
            self.method = method

        def validate_shape(self, layer):
            if layer.input_size_per_partition % 32 or layer.output_size_per_partition % 32:
                raise ValueError('integer adapter requires N and K divisible by 32')

        def finalize_prefill(self, layer):
            for name in self.names:
                parameter = getattr(layer, f'ph_prefill_{name}')
                parameter.data = parameter.data.contiguous()
            if self.method == 'marlin_int4':
                from standalone_marlin_int4.marlin_int4.linear import _make_workspace
                layer.ph_prefill_weight_scale.data=layer.ph_prefill_weight_scale.data.to(torch.bfloat16)
                layer.ph_prefill_int4_workspace=_make_workspace(layer.ph_prefill_packed_weight.device,layer.output_size_per_partition)

        def finalize_decode(self, layer):
            for name in self.names:
                key = f'ph_decode_{name}_gpu'
                setattr(layer, key, getattr(layer, key).contiguous())
            if self.method == 'marlin_int4':
                from standalone_marlin_int4.marlin_int4.linear import _make_workspace
                layer.ph_decode_weight_scale_gpu=layer.ph_decode_weight_scale_gpu.to(torch.bfloat16)
                layer.ph_decode_int4_workspace=_make_workspace(layer.ph_decode_packed_weight_gpu.device,layer.output_size_per_partition)

        def apply(self, layer, x, bias, decode):
            prefix, suffix = ('ph_decode_', '_gpu') if decode else ('ph_prefill_', '')
            weight = getattr(layer, prefix + 'packed_weight' + suffix)
            scale = getattr(layer, prefix + 'weight_scale' + suffix)
            if self.method == 'marlin_int4':
                from int4_marlin import forward
                workspace=(layer.ph_prefill_int4_workspace if not decode or layer.ph_prefill_method==layer.ph_decode_method
                           else layer.ph_decode_int4_workspace)
                out=forward(x,weight,scale,workspace,layer.output_size_per_partition,layer.input_size_per_partition)
                return out if bias is None else out+bias
            if self.method == 'int8':
                out = int8_forward(x, weight, scale, backend='int_mm')
                return out if bias is None else out + bias
            from cutlass_wrapper import pack_w4a4_int4_activation, w4a4_int4_gemm
            shape = x.shape[:-1]
            flat = x.reshape(-1, layer.input_size_per_partition).contiguous()
            activation, activation_scale = pack_w4a4_int4_activation(flat)
            out = w4a4_int4_gemm(activation, activation_scale, weight, scale, bias,
                flat.shape[0], layer.output_size_per_partition, layer.input_size_per_partition)
            return out.reshape(*shape, layer.output_size_per_partition)

    for method in ['int8', 'int4', 'marlin_int4']:
        phase.SUPPORTED_METHODS.add(method)
        phase.ADAPTERS[method] = IntegerAdapter(method)

    class EightBitAdapter(phase.LinearAdapter):
        names = ('packed_weight', 'weight_scale', 'input_smooth')

        def __init__(self, method):
            self.method = method

        def validate_shape(self, layer):
            if layer.input_size_per_partition % 32 or layer.output_size_per_partition % 32:
                raise ValueError('8bit GEMM requires N,K divisible by 32')

        def apply(self, layer, x, bias, decode):
            from eightbit import forward
            prefix, suffix = ('ph_decode_', '_gpu') if decode else ('ph_prefill_', '')
            tensors = [getattr(layer, prefix + name + suffix) for name in self.names]
            out = forward(x, *tensors, self.method)
            return out if bias is None else out + bias

    for method in ['sq_int8', 'fp8', 'sq_fp8']:
        phase.SUPPORTED_METHODS.add(method)
        phase.ADAPTERS[method] = EightBitAdapter(method)
    return phase
