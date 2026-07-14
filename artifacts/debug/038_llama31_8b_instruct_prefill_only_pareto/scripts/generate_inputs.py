#!/usr/bin/env python3
"""Freeze Llama3-tokenized WikiText blocks and the controlled quality policies."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct')
METHODS = ('dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4', 'w4a16_ours')
TYPES = ('qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj')


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocks', type=int, default=100)
    p.add_argument('--seed', type=int, default=86)
    p.add_argument('--seq-len', type=int, default=2048)
    return p.parse_args()


def method_map(index):
    result = {}
    for layer in range(32):
        bucket = layer // 8
        for typ in TYPES:
            if index < 5:
                method = METHODS[index]
            elif index < 21:
                cell = index - 5
                selected_bucket, selected_type = cell // 4, TYPES[cell % 4]
                method = METHODS[1 + (cell % 4)] if (bucket, typ) == (selected_bucket, selected_type) else 'dense_bf16'
            elif index < 37:
                # A second controlled family changes a whole bucket, improving
                # coverage of interactions without leaking holdout labels.
                bucket_to_change, method_index = divmod(index - 21, 4)
                method = METHODS[1 + method_index] if bucket == bucket_to_change else 'dense_bf16'
            else:
                # The first version used a periodic arithmetic expression,
                # accidentally creating only five repeated "mixed" policies.
                # Use independent, deterministic placements at five severity
                # levels instead, so both train and frozen holdout cover the
                # mixed-policy distribution.
                severity = (index - 37) % 5
                probabilities = (
                    (0.75, 0.10, 0.03, 0.02, 0.10),
                    (0.55, 0.18, 0.07, 0.04, 0.16),
                    (0.38, 0.25, 0.09, 0.06, 0.22),
                    (0.22, 0.30, 0.13, 0.10, 0.25),
                    (0.08, 0.30, 0.18, 0.16, 0.28),
                )[severity]
                rng = random.Random(1307 + index * 100003 + layer * 97 + TYPES.index(typ))
                draw, total = rng.random(), 0.0
                method = METHODS[-1]
                for candidate, probability in zip(METHODS, probabilities):
                    total += probability
                    if draw <= total:
                        method = candidate
                        break
            result[f'model.layers.{layer}.self_attn.{typ}' if typ in ('qkv_proj', 'o_proj') else f'model.layers.{layer}.mlp.{typ}'] = {'prefill_method': method, 'decode_method': method}
    return result


def main():
    a = parse()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', cache_dir='/home/agent/wja/.cache/huggingface')
    tokens = tokenizer('\n\n'.join(data['text']), return_tensors='pt', add_special_tokens=False).input_ids[0]
    if len(tokens) < a.blocks * a.seq_len:
        raise RuntimeError(f'only {len(tokens)} tokens available')
    g = torch.Generator().manual_seed(a.seed)
    starts = torch.randint(0, len(tokens) - a.seq_len - 1, (a.blocks,), generator=g).tolist()
    blocks = torch.stack([tokens[s:s + a.seq_len + 1] for s in starts])
    samples = ROOT / 'samples'; samples.mkdir(parents=True, exist_ok=True)
    torch.save(blocks, samples / 'wikitext_2048.pt')
    (samples / 'metadata.json').write_text(json.dumps({'dataset': 'Salesforce/wikitext:wikitext-2-raw-v1/train', 'model': str(MODEL), 'blocks': a.blocks, 'seq_len': a.seq_len, 'seed': a.seed, 'starts': starts}, indent=2) + '\n')
    directory = ROOT / 'policies' / 'prefill_only'; directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index in range(72):
        policy_id = f'p{index:02d}'
        policy = {'policy_id': policy_id, 'scenario': 'prefill_only', 'policy_kind': 'controlled' if index < 37 else 'balanced_mixed', 'default_prefill_method': 'dense_bf16', 'default_decode_method': 'dense_bf16', 'modules_to_not_convert': ['lm_head'], 'method_map': method_map(index)}
        path = directory / f'{policy_id}.json'; path.write_text(json.dumps(policy, indent=2, sort_keys=True) + '\n')
        manifest.append({'policy_id': policy_id, 'split': 'train' if index < 54 else 'holdout', 'policy_kind': policy['policy_kind'], 'path': str(path)})
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'samples': tuple(blocks.shape), 'policies': len(manifest), 'train': 54, 'holdout': 18}, indent=2))


if __name__ == '__main__':
    main()
