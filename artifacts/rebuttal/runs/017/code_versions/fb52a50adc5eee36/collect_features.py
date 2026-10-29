"""Measure target-model SQ/FP8/plain INT8; preserve original bucket features."""
import csv
import gc
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from common import MODEL, RUN, LAYERS, Int8Linear, write_json
from eightbit import EightBitLinear
from graph_bindings import install
install()
from cutlass_wrapper.canonical_nvfp4 import quantize_nvfp4_canonical_weight, dequantize_nvfp4_canonical_weight

SPECS = [('self_attn.q_proj', 'self_attn.qkv_proj', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
         ('self_attn.o_proj', 'self_attn.o_proj', ['self_attn.o_proj']),
         ('mlp.gate_proj', 'mlp.gate_up_proj', ['mlp.gate_proj', 'mlp.up_proj']),
         ('mlp.down_proj', 'mlp.down_proj', ['mlp.down_proj'])]
NEW = ['sq_int8', 'fp8', 'int8']


class CapturedInput(Exception):
    pass


@torch.inference_mode()
def main():
    started = time.perf_counter()
    assert not (RUN / 'results/features_complete.json').exists()
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
    scales, profiles, rows, weights = {}, {}, [], []
    for index, layer in enumerate(model.model.layers):
        maxima, handles, outputs = {}, [], []
        for hook, kind, _parts in SPECS:
            def collect(_module, args, _out, kind=kind):
                value = args[0][:, :2048].abs().reshape(-1, args[0].shape[-1]).amax(0)
                maxima[kind] = value if kind not in maxima else torch.maximum(maxima[kind], value)
            handles.append(layer.get_submodule(hook).register_forward_hook(collect))
        for hidden in inputs:
            result = layer(hidden.to('cuda'), **kwargs)
            outputs.append((result[0] if isinstance(result, tuple) else result).cpu())
        for handle in handles:
            handle.remove()
        for kind, value in maxima.items():
            scales[f'model.layers.{index}.{kind}'] = value.cpu()
        references, compressed, accum, handles = {}, {}, defaultdict(lambda: [0., 0., 0]), []
        for hook, kind, parts in SPECS:
            name = f'model.layers.{index}.{kind}'
            weight = torch.cat([layer.get_submodule(part).weight for part in parts])
            references[kind] = weight
            if index == 0:
                profiles[name] = weight.cpu()
            for method in NEW:
                compressed[kind, method] = (Int8Linear(weight, backend='int_mm') if method == 'int8'
                    else EightBitLinear(method, name, weight, scales)).eval()
            integer = compressed[kind, 'int8']
            decoded = {'int8': integer.weight.float()*integer.scale,
                'dense_nvfp4': dequantize_nvfp4_canonical_weight(quantize_nvfp4_canonical_weight(weight.contiguous()))}
            offset = 0
            for part in parts:
                size = layer.get_submodule(part).weight.shape[0]
                for method, effective in decoded.items():
                    delta = effective[offset:offset+size].double()-weight[offset:offset+size].double()
                    weights.append({'module': f'model.layers.{index}.{part}', 'fused_module': name,
                        'method': method, 'weight_error_squared': float(delta.square().sum())})
                offset += size
            del decoded, effective, delta

            def error(_module, args, _out, kind=kind):
                for phase, x in [('prefill', args[0][:, :2048]), ('decode', args[0][:, 2048:2112])]:
                    reference = F.linear(x, references[kind]).float()
                    mass = float(reference.square().sum())
                    for method in NEW:
                        output = compressed[kind, method](x).float()
                        value = accum[phase, kind, method]
                        value[0] += float((output-reference).square().sum())
                        value[1] += mass
                        value[2] += reference.numel()
            handles.append(layer.get_submodule(hook).register_forward_hook(error))
        for hidden in inputs[:16]:
            layer(hidden.to('cuda'), **kwargs)
        for handle in handles:
            handle.remove()
        for (phase, kind, method), (sse, mass, count) in accum.items():
            rows.append({'module': f'model.layers.{index}.{kind}', 'phase': phase, 'method': method,
                'layer_bucket': index*4//LAYERS, 'fused_type': kind.rsplit('.', 1)[-1],
                'sse': sse, 'reference_mass': mass, 'output_count': count,
                'output_rel_mse': sse/max(mass, 1e-12)})
        inputs = outputs
        del compressed, references, handles, weight
        gc.collect()
        torch.cuda.empty_cache()
        print('Target-model 8bit features', index+1, '/', LAYERS, flush=True)
    assert len(rows) == LAYERS*4*2*3 and len(weights) == LAYERS*7*2
    for filename, data in [('local_errors_new_module.csv', rows), ('weight_errors_runtime.csv', weights)]:
        with (RUN / 'data' / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    aggregate = defaultdict(lambda: [0., 0.])
    for row in rows:
        target = aggregate[row['phase'], row['layer_bucket'], row['fused_type'], row['method']]
        target[0] += row['sse']
        target[1] += row['reference_mass']
    for scene in ['prefill_only', 'prefill_decode']:
        values = {key: sse/max(mass, 1e-12) for key, (sse, mass) in aggregate.items()}
        for path in (RUN / f'reference/local_errors/{scene}').glob('*.csv'):
            for row in csv.DictReader(path.open()):
                values[row['phase'], int(row['layer_bucket']), row['fused_type'], row['method']] = float(row['output_rel_mse'])
        projected = [{'module': f'model.layers.{i}.{kind}', 'phase': phase, 'method': method,
            'output_rel_mse': values[phase, i*4//LAYERS, kind.rsplit('.', 1)[-1], method]}
            for i in range(LAYERS) for _, kind, _ in SPECS for phase in ['prefill', 'decode']
            for method in ['dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours',*NEW]]
        with (RUN / f'data/local_errors_{scene}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(projected[0]))
            writer.writeheader()
            writer.writerows(projected)
    torch.save(scales, RUN / 'data/activation_scales.pt')
    torch.save(profiles, RUN / 'data/profile_weights.pt')
    write_json(RUN / 'results/features_complete.json', {'complete': True, 'seconds': time.perf_counter()-started,
        'new_module_rows': len(rows), 'weight_error_rows': len(weights), 'scale_contexts': 128,
        'error_contexts': 16, 'smoothing_alpha': .5,
        'scope': 'Original per-depth/type bucket features retained for five actions; target-model new methods measured and aggregated identically. Dense backbone inputs shared, continuation M64 feature approximation disclosed.'})


if __name__ == '__main__':
    main()
