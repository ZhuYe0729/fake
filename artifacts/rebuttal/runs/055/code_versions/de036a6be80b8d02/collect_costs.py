"""Real fused-module perturbations with dense-model inputs and canonical states."""
from __future__ import annotations

import csv
import gc
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from common import MODEL, RUN, compressed_module, write_json
from prepare_sparse import CapturedInput


@torch.inference_mode()
def main():
    from graph_bindings import install
    install()
    started = time.perf_counter()
    methods = ['dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4', 'w4a16_ours']
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='eager').to('cuda').eval()
    blocks = torch.load(RUN / 'data/calibration_128_2112.pt', weights_only=True)[:16]
    states = {method: torch.load(RUN / 'data/canonical' / method / 'model.pt',
                map_location='cpu', mmap=True, weights_only=True)['state_dict']
              for method in ['sparse_bf16', 'sparse_nvfp4']}
    inputs, kwargs = [], {}

    def capture(_layer, args, keywords):
        inputs.append(args[0].cpu())
        kwargs.update(keywords)
        raise CapturedInput

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for tokens in blocks:
            try:
                model(tokens[None].to('cuda'), use_cache=False)
            except CapturedInput:
                pass
    finally:
        handle.remove()
    rows = []
    specs = [('self_attn.q_proj', 'qkv_proj', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
             ('self_attn.o_proj', 'o_proj', ['self_attn.o_proj']),
             ('mlp.gate_proj', 'gate_up_proj', ['mlp.gate_proj', 'mlp.up_proj']),
             ('mlp.down_proj', 'down_proj', ['mlp.down_proj'])]
    for index, layer in enumerate(model.model.layers):
        references, compressed, handles = {}, {}, []
        accum = defaultdict(lambda: [0.0, 0.0, 0])
        for hook_name, kind, parts in specs:
            raw = torch.cat([layer.get_submodule(name).weight for name in parts])
            references[kind] = raw
            for method in methods:
                weight = torch.cat([states[method][f'model.layers.{index}.{part}.weight'] for part in parts]).to('cuda') if method in states else raw
                compressed[kind, method] = compressed_module(method, weight).eval()

            def hook(_module, args, _output, *, kind=kind):
                full = args[0]
                for phase, x in [('prefill', full[:, :2048]), ('decode', full[:, 2048:2112])]:
                    reference = F.linear(x, references[kind]).float()
                    mass = float(reference.square().sum())
                    for method in methods:
                        output = compressed[kind, method](x).float()
                        row = accum[phase, kind, method]
                        row[0] += float((output - reference).square().sum())
                        row[1] += mass
                        row[2] += reference.numel()
            handles.append(layer.get_submodule(hook_name).register_forward_hook(hook))
        next_inputs = []
        try:
            for hidden in inputs:
                output = layer(hidden.to('cuda'), **kwargs)
                next_inputs.append((output[0] if isinstance(output, tuple) else output).cpu())
        finally:
            for handle in handles:
                handle.remove()
        for (phase, kind, method), (sse, mass, count) in accum.items():
            group = 'self_attn' if kind in ['qkv_proj', 'o_proj'] else 'mlp'
            rows.append({'module': f'model.layers.{index}.{group}.{kind}', 'layer': index,
                'phase': phase, 'method': method, 'layer_bucket': index * 4 // model.config.num_hidden_layers,
                'fused_type': kind, 'sse': sse, 'reference_mass': mass,
                'output_count': count, 'output_rel_mse': sse / max(mass, 1e-12)})
        inputs = next_inputs
        del compressed, references, handles, raw, weight
        gc.collect()
        torch.cuda.empty_cache()
        print('Local costs layer', index + 1, 'seconds', time.perf_counter() - started, flush=True)
        with (RUN / 'data/local_errors_five.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(RUN / 'results/local_error_metadata.json', {
        'blocks': len(blocks), 'input_tokens': 2048, 'decode_tokens': 64,
        'depth_bucket': 'floor(layer * 4 / 64)',
        'reference': 'identical dense-model inputs, fused BF16 Linear outputs',
        'compressed': 'same canonical states and packed runtime modules as exported policies',
        'elapsed_seconds': time.perf_counter() - started, 'rows': len(rows)})


if __name__ == '__main__':
    main()
