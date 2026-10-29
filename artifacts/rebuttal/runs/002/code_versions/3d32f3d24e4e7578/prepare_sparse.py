"""Sequential-layer SparseGPT; same pruning implementation as DASO canonical states."""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from common import MODEL, RUN, write_json
from daso.compression import prune_weight


class CapturedInput(Exception):
    pass


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['sparse_bf16', 'sparse_nvfp4'], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=128)
    args = parser.parse_args()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='eager').to('cuda').eval()
    blocks = torch.load(RUN / 'data/calibration_128_2112.pt', weights_only=True)[:args.blocks, :2048]
    inputs = []
    layer_kwargs = {}

    def capture(_layer, positional, kwargs):
        inputs.append(positional[0].detach().cpu())
        layer_kwargs.update(kwargs)
        raise CapturedInput

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for block in blocks:
            try:
                model(block[None].to('cuda'), use_cache=False)
            except CapturedInput:
                pass
    finally:
        handle.remove()
    if len(inputs) != len(blocks):
        raise RuntimeError('failed to capture every calibration block')
    names = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    records = []
    pattern = 'dense_2_4' if args.method == 'sparse_bf16' else 'nvfp4_pair_2_4_over_8'
    for index, layer in enumerate(model.model.layers):
        linears = [(name, module) for name, module in layer.named_modules()
                   if name.rsplit('.', 1)[-1] in names]
        hessians = {name: torch.zeros((module.in_features, module.in_features),
                    dtype=torch.float32, device='cuda') for name, module in linears}
        counts = {name: 0 for name, _ in linears}
        handles = []
        for name, module in linears:
            def hook(_module, args, _out, *, name=name):
                x = args[0].detach().float().reshape(-1, args[0].shape[-1])
                old = counts[name]
                counts[name] += len(x)
                hessians[name].mul_(old / counts[name])
                hessians[name].addmm_(x.T, x, alpha=2.0 / counts[name])
            handles.append(module.register_forward_hook(hook))
        try:
            for hidden in inputs:
                layer(hidden.to('cuda'), **layer_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        for name, module in linears:
            full_name = f'model.layers.{index}.{name}'
            weight, _mask, metadata = prune_weight(module.weight, hessians[name],
                pattern=pattern, block_size=128, percdamp=.01, module_name=full_name)
            module.weight.copy_(weight)
            records.append({'module': full_name, **metadata})
        del hessians
        next_inputs = []
        for hidden in inputs:
            output = layer(hidden.to('cuda'), **layer_kwargs)
            next_inputs.append((output[0] if isinstance(output, tuple) else output).cpu())
        inputs = next_inputs
        gc.collect()
        torch.cuda.empty_cache()
        print(args.method, 'layer', index + 1, 'elapsed', time.perf_counter() - started, flush=True)
    target = args.output_root / args.method / 'model.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'algorithm': 'sparsegpt', 'pattern': pattern, 'quantized': False,
                'model_path': str(MODEL), 'sample_count': len(blocks),
                'block_size': 128, 'percdamp': .01, 'modules': records,
                'elapsed_seconds': time.perf_counter() - started,
                'execution': 'cached sequential decoder layers'}
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({'state_dict': state, 'metadata': metadata}, target)
    write_json(RUN / f'results/canonical_{args.method}.json', metadata)
    print('Saved', target, flush=True)


if __name__ == '__main__':
    main()
