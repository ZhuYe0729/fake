#!/usr/bin/env python3
"""Measure teacher-forced prefill WikiText NLL for one frozen Llama3 policy."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct')
PREPARED = Path('/home/agent/wja/project/my/cospaq/fake/artifacts/exports/vllm/baselines/llama3.1-8b-instruct/prepared')
STATE = {'dense_nvfp4': 'dense_nvfp4', 'sparse_bf16': 'sparse_bf16', 'sparse_nvfp4': 'sparse_nvfp4', 'w4a16_ours': 'marlin_nvfp4'}


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True)
    p.add_argument('--policy-json', type=Path)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--blocks', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--dense-reference-json', type=Path)
    p.add_argument('--output-csv', type=Path, required=True)
    return p.parse_args()


def parent(model, name):
    obj = model
    for part in name.split('.')[:-1]:
        obj = getattr(obj, part)
    return obj, name.rsplit('.', 1)[-1]


def sources(fused):
    base, typ = fused.rsplit('.', 1)
    if typ == 'qkv_proj': return [base + '.q_proj', base + '.k_proj', base + '.v_proj']
    if typ == 'gate_up_proj': return [base + '.gate_proj', base + '.up_proj']
    return [fused]


def install(model, policy):
    for method, artifact in STATE.items():
        selected = [name for name, item in policy['method_map'].items() if item['prefill_method'] == method]
        if not selected: continue
        state = torch.load(PREPARED / artifact / 'model.pt', map_location='cpu')['state_dict']
        for fused in selected:
            for name in sources(fused):
                obj, child = parent(model, name); old = getattr(obj, child)
                # One policy is evaluated per process.  In-place overwrite
                # avoids retaining a second full 8B model solely for restore.
                old.weight.data.copy_(state[f'{name}.weight'].to(old.weight))
        del state
    gc.collect()
    return None


@torch.inference_mode()
def nll(model, blocks, device, batch_size):
    total, tokens = 0.0, 0
    for start in range(0, len(blocks), batch_size):
        ids = blocks[start:start + batch_size].to(device)
        logits = model(input_ids=ids[:, :-1], use_cache=False).logits.float()
        labels = ids[:, 1:]
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='sum').item())
        tokens += int(labels.numel())
    return total / tokens, tokens


def main():
    a = parse(); device = f'cuda:{a.gpu}'; torch.cuda.set_device(a.gpu)
    blocks = torch.load(ROOT / 'samples/wikitext_2048.pt', map_location='cpu')[:a.blocks]
    policy_path = a.policy_json or (ROOT / 'policies/prefill_only' / f'{a.policy}.json')
    policy = json.loads(policy_path.read_text())
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation='eager').to(device).eval()
    if a.dense_reference_json and a.dense_reference_json.exists():
        if a.dense_reference_json.suffix == '.csv':
            reference = next(csv.DictReader(a.dense_reference_json.open()))
        else:
            reference = json.loads(a.dense_reference_json.read_text())
        dense_nll, tokens = float(reference['dense_prefill_nll']), int(reference['tokens'])
    else:
        dense_nll, tokens = nll(model, blocks, device, a.batch_size)
    install(model, policy)
    measured, _ = nll(model, blocks, device, a.batch_size)
    del model; gc.collect(); torch.cuda.empty_cache()
    row = {'policy_id': a.policy, 'scenario': 'prefill_only', 'sample_count': len(blocks), 'tokens': tokens,
           'dense_prefill_nll': dense_nll, 'prefill_nll': measured, 'delta_prefill_nll': measured - dense_nll,
           'target_delta_nll': measured - dense_nll}
    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.output_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
