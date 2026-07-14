#!/usr/bin/env python3
"""Collect phase-specific Llama3 local output-relative-MSE features."""
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct')
PREPARED = Path('/home/agent/wja/project/my/cospaq/fake/artifacts/exports/vllm/baselines/llama3.1-8b-instruct/prepared')
ARTIFACT = {'dense_nvfp4': 'dense_nvfp4', 'sparse_bf16': 'sparse_bf16', 'sparse_nvfp4': 'sparse_nvfp4', 'w4a16_ours': 'marlin_nvfp4'}
LINEARS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=tuple(ARTIFACT), required=True)
    p.add_argument('--phase', choices=('prefill', 'decode'), required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--blocks', type=int, default=16)
    p.add_argument('--module-chunk-size', type=int, default=8)
    return p.parse_args()


def bucket(name): return int(name.split('.')[2]) // 8
def fused_type(name):
    typ = name.rsplit('.', 1)[-1]
    return 'qkv_proj' if typ in {'q_proj', 'k_proj', 'v_proj'} else 'gate_up_proj' if typ in {'gate_proj', 'up_proj'} else typ


def main():
    a = parse(); torch.cuda.set_device(a.gpu); device = f'cuda:{a.gpu}'
    blocks = torch.load(ROOT / 'samples/wikitext_2048.pt', map_location='cpu')[:a.blocks]
    state = torch.load(PREPARED / ARTIFACT[a.method] / 'model.pt', map_location='cpu')['state_dict']
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation='eager').to(device).eval()
    acc = {}
    def hook(name):
        def apply(module, inputs, output):
            x, y = inputs[0], output
            if a.phase == 'decode': x, y = x[:, -80:], y[:, -80:]
            weight = state[f'{name}.weight'].to(device=x.device, dtype=x.dtype, non_blocking=True)
            hat = F.linear(x, weight, module.bias); error = (hat.float() - y.float())
            row = acc.setdefault((bucket(name), fused_type(name)), {'sse': 0., 'ref': 0., 'count': 0})
            row['sse'] += float(error.square().sum()); row['ref'] += float(y.float().square().sum()); row['count'] += int(y.numel())
        return apply
    modules = [(n, m) for n, m in model.named_modules() if n.rsplit('.', 1)[-1] in LINEARS and f'{n}.weight' in state]
    for start in range(0, len(modules), a.module_chunk_size):
        handles = [m.register_forward_hook(hook(n)) for n, m in modules[start:start + a.module_chunk_size]]
        try:
            with torch.inference_mode():
                for block in blocks: model(input_ids=block[:-1].unsqueeze(0).to(device), use_cache=False)
        finally:
            for handle in handles: handle.remove()
            torch.cuda.empty_cache(); gc.collect()
    rows = [{'method': a.method, 'layer_bucket': b, 'fused_type': typ, 'blocks': a.blocks,
             'output_rel_mse': v['sse'] / max(v['ref'], 1e-12), 'output_mse': v['sse'] / max(v['count'], 1), 'output_count': v['count']}
            for (b, typ), v in sorted(acc.items())]
    out = ROOT / 'local_errors' / f'{a.phase}_{a.method}.csv'; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == '__main__':
    main()
