"""Add genuine plain-INT8 local output errors using the original BF16 contexts."""
import csv
import gc
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from common import MODEL, RUN, Int8Linear, write_json
from graph_bindings import install
install()
from cutlass_wrapper.canonical_nvfp4 import quantize_nvfp4_canonical_weight, dequantize_nvfp4_canonical_weight

SPECS = [('self_attn.q_proj', 'self_attn.qkv_proj', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
         ('self_attn.o_proj', 'self_attn.o_proj', ['self_attn.o_proj']),
         ('mlp.gate_proj', 'mlp.gate_up_proj', ['mlp.gate_proj', 'mlp.up_proj']),
         ('mlp.down_proj', 'mlp.down_proj', ['mlp.down_proj'])]


class CapturedInput(Exception):
    pass


@torch.inference_mode()
def main():
    started = time.perf_counter()
    assert not (RUN / 'data/local_errors_extended.csv').exists()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='eager').to('cuda').eval()
    tokens = torch.load(RUN / 'data/calibration_128_2112.pt', weights_only=True)
    inputs, kwargs = [], {}

    def capture(_layer, args, keywords):
        inputs.append(args[0].cpu())
        kwargs.update(keywords)
        raise CapturedInput

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    for block in tokens:
        try:
            model(block[None].to('cuda'), use_cache=False)
        except CapturedInput:
            pass
    handle.remove()
    with (RUN / 'data/local_errors.csv').open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    weight_rows = []
    for index, layer in enumerate(model.model.layers):
        references, compressed, handles = {}, {}, []
        accum = defaultdict(lambda: [0., 0., 0])
        for hook, kind, parts in SPECS:
            weight = torch.cat([layer.get_submodule(part).weight for part in parts])
            references[kind] = weight
            compressed[kind] = Int8Linear(weight, backend='int_mm').eval()
            decoded = {'int8': compressed[kind].weight.float() * compressed[kind].scale,
                       'dense_nvfp4': dequantize_nvfp4_canonical_weight(quantize_nvfp4_canonical_weight(weight.contiguous()))}
            offset = 0
            for part in parts:
                size = layer.get_submodule(part).weight.shape[0]
                for method, effective in decoded.items():
                    delta = effective[offset:offset+size].double() - weight[offset:offset+size].double()
                    weight_rows.append({'module': f'model.layers.{index}.{part}',
                        'fused_module': f'model.layers.{index}.{kind}', 'method': method,
                        'weight_error_squared': float(delta.square().sum())})
                offset += size
            del decoded, effective, delta

            def error(_module, args, _out, kind=kind):
                for phase, x in [('prefill', args[0][:, :2048]), ('decode', args[0][:, 2048:2112])]:
                    reference = F.linear(x, references[kind]).float()
                    output = compressed[kind](x).float()
                    value = accum[phase, kind]
                    value[0] += float((output-reference).square().sum())
                    value[1] += float(reference.square().sum())
                    value[2] += reference.numel()
            handles.append(layer.get_submodule(hook).register_forward_hook(error))
        outputs = []
        for block, hidden in enumerate(inputs):
            if block == 16:
                for handle in handles:
                    handle.remove()
            result = layer(hidden.to('cuda'), **kwargs)
            outputs.append((result[0] if isinstance(result, tuple) else result).cpu())
        for handle in handles:
            handle.remove()
        for (phase, kind), (sse, mass, count) in accum.items():
            rows.append({'module': f'model.layers.{index}.{kind}', 'layer': index, 'phase': phase,
                'method': 'int8', 'layer_bucket': index*4//36, 'fused_type': kind.rsplit('.', 1)[-1],
                'sse': sse, 'reference_mass': mass, 'output_count': count, 'output_rel_mse': sse/max(mass, 1e-12)})
        inputs = outputs
        del references, compressed, weight, handles
        gc.collect()
        torch.cuda.empty_cache()
        print('plain INT8 local errors layer', index, flush=True)
    assert sum(row['method'] == 'int8' for row in rows) == 288
    with (RUN / 'data/local_errors_extended.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    assert len(weight_rows) == 504
    with (RUN / 'data/weight_errors_runtime.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)
    write_json(RUN / 'results/int8_local_errors_complete.json', {'complete': True, 'new_rows': 288,
        'seconds': time.perf_counter()-started, 'samples': 16,
        'scope': 'BF16 propagated train contexts; actual plain INT8 quantizer and runtime torch._int_mm/Triton epilogue with identity smoothing, no SQ-cost substitution.'})


if __name__ == '__main__':
    main()
