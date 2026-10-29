from __future__ import annotations

import importlib.util
import sys

import torch

from common import VLLM, quantize_int8


def main() -> None:
    path = VLLM / 'artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py'
    spec = importlib.util.spec_from_file_location('original_phase_exporter', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.SUPPORTED_METHODS.update(['int8', 'int4', 'marlin_int4'])
    convert = original.convert_method

    @torch.no_grad()
    def convert_method(method, name, weight, prune):
        if method == 'int8':
            quantized, scale = quantize_int8(weight.to('cuda', dtype=torch.bfloat16))
            return {f'{name}.packed_weight': quantized.cpu(), f'{name}.weight_scale': scale.cpu()}
        if method == 'int4':
            from cutlass_wrapper import quantize_w4a4_int4_weight
            packed = quantize_w4a4_int4_weight(weight.to('cuda', dtype=torch.bfloat16).contiguous())
            return {f'{name}.packed_weight': packed.packed_weight.cpu(),
                    f'{name}.weight_scale': packed.weight_scale.cpu()}
        if method == 'marlin_int4':
            from int4_marlin import pack
            packed,scale,_=pack(weight.to('cuda',dtype=torch.bfloat16).contiguous())
            return {f'{name}.packed_weight':packed.cpu(),f'{name}.weight_scale':scale.cpu()}
        return convert(method, name, weight, prune)

    original.convert_method = convert_method
    original.main()


if __name__ == '__main__':
    main()
