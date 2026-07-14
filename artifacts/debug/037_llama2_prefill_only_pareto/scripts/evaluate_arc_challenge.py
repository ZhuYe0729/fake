#!/usr/bin/env python3
"""Evaluate one prefill-only policy on ARC-Challenge by answer likelihood."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL = Path('/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf')
PREPARED = Path('/home/agent/wja/project/my/cospaq/fake/artifacts/exports/vllm/baselines/llama2-7b-chat/prepared')
STATE = {
    'dense_nvfp4': 'dense_nvfp4',
    'sparse_bf16': 'sparse_bf16',
    'sparse_nvfp4': 'sparse_nvfp4',
    'w4a16_ours': 'marlin_nvfp4',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-json', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batch-size', default='4')
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    obj = model
    for part in name.split('.')[:-1]:
        obj = getattr(obj, part)
    return obj, name.rsplit('.', 1)[-1]


def sources(fused: str) -> list[str]:
    base, kind = fused.rsplit('.', 1)
    if kind == 'qkv_proj':
        return [base + '.q_proj', base + '.k_proj', base + '.v_proj']
    if kind == 'gate_up_proj':
        return [base + '.gate_proj', base + '.up_proj']
    return [fused]


def install_prefill_policy(model: nn.Module, policy: dict) -> list[tuple[nn.Module, str, nn.Module]]:
    saved = []
    for method, artifact in STATE.items():
        selected = [
            name for name, entry in policy['method_map'].items()
            if entry['prefill_method'] == method
        ]
        if not selected:
            continue
        state = torch.load(PREPARED / artifact / 'model.pt', map_location='cpu')['state_dict']
        for fused in selected:
            for name in sources(fused):
                obj, child = parent(model, name)
                old = getattr(obj, child)
                new = nn.Linear(
                    old.in_features, old.out_features, bias=old.bias is not None,
                    device=old.weight.device, dtype=old.weight.dtype,
                )
                new.weight.data.copy_(state[f'{name}.weight'].to(old.weight))
                if old.bias is not None:
                    new.bias.data.copy_(old.bias.data)
                setattr(obj, child, new)
                saved.append((obj, child, old))
        del state
        gc.collect()
    return saved


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = f'cuda:{args.gpu}'
    policy = json.loads(args.policy_json.read_text())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='eager',
    ).to(device).eval()
    saved = install_prefill_policy(model, policy)
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(
            pretrained=model, tokenizer=str(MODEL), backend='causal', dtype=torch.bfloat16,
            device=device, batch_size=args.batch_size, trust_remote_code=False,
        )
        result = lm_eval.simple_evaluate(
            model=lm, tasks=['arc_challenge'], num_fewshot=0,
            batch_size=args.batch_size, limit=args.limit, log_samples=False,
        )
        if result is None:
            raise RuntimeError('lm_eval.simple_evaluate returned None')
        metrics = result['results']['arc_challenge']
        row = {
            'label': args.label,
            'policy_json': str(args.policy_json),
            'limit': args.limit,
            'batch_size': args.batch_size,
            'acc': metrics.get('acc,none'),
            'acc_norm': metrics.get('acc_norm,none'),
            'raw_metrics': metrics,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        for obj, child, old in saved:
            setattr(obj, child, old)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
