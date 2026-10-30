"""HAWQ weight errors from the actual fused FP4/INT8 quantization groups."""
import csv
from contextlib import ExitStack
import json

from safetensors import safe_open
import torch

from common import MODEL, RUN, quantize_int8, write_json
from cutlass_wrapper.canonical_nvfp4 import quantize_nvfp4_canonical_weight, dequantize_nvfp4_canonical_weight

PARTS = {'self_attn.qkv_proj': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
         'self_attn.o_proj': ['self_attn.o_proj'],
         'mlp.gate_up_proj': ['mlp.gate_proj', 'mlp.up_proj'],
         'mlp.down_proj': ['mlp.down_proj']}


@torch.inference_mode()
def main():
    target = RUN / 'results/weight_quantization_errors.csv'
    assert not target.exists()
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    rows = []
    with ExitStack() as stack:
        readers = {name: stack.enter_context(safe_open(MODEL / name, framework='pt', device='cpu'))
                   for name in set(index.values())}
        for layer in range(64):
            for fused, parts in PARTS.items():
                names = [f'model.layers.{layer}.{part}.weight' for part in parts]
                tensors = [readers[index[name]].get_tensor(name) for name in names]
                weight = torch.cat(tensors).cuda()
                packed = quantize_nvfp4_canonical_weight(weight)
                fp4 = dequantize_nvfp4_canonical_weight(packed)
                quantized, scale = quantize_int8(weight)
                integer = quantized.float() * scale
                offset = 0
                for name, tensor in zip(names, tensors, strict=True):
                    end = offset + tensor.shape[0]
                    reference = weight[offset:end].float()
                    for method, output in [('dense_nvfp4', fp4), ('int8', integer), ('dense_bf16', None)]:
                        rows.append({'module': name.removesuffix('.weight'),
                            'fused_module': f'model.layers.{layer}.{fused}', 'method': method,
                            'numel': tensor.numel(), 'weight_error_squared': 0. if output is None else
                            float((reference - output[offset:end]).square().sum())})
                    offset = end
                del packed, fp4, quantized, scale, integer, weight, reference, output
            print('weight errors layer', layer + 1, flush=True)
    assert len(rows) == 64 * 7 * 3
    with target.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(RUN / 'results/weight_errors_complete.json', {'complete': True, 'rows': len(rows),
        'scope': 'Quantize each runtime-fused matrix once; partition its weight error back to raw projection Hessian blocks. '
                 'Weight-only HAWQ statistic, without claiming activation error is modeled.'})


if __name__ == '__main__':
    main()
