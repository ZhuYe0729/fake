#!/usr/bin/env python3
"""Evaluate one prefill-only policy on ARC-Challenge by answer likelihood."""
from __future__ import annotations

import argparse
import gc
import json
import os
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


def install_local_arc_dataset() -> callable:
    """Route lm-eval's ARC request to the immutable local Arrow files.

    The cache was produced by an older datasets release, whose ``List``
    metadata now conflicts with datasets 3.x cache reconstruction.  Reading
    the Arrow splits directly preserves the exact ARC data and avoids cache
    locks and schema reconstruction altogether.
    """
    import datasets
    from datasets import Dataset, DatasetDict
    from datasets.features import features

    if "List" not in features._FEATURE_TYPES:
        features._FEATURE_TYPES["List"] = features.Sequence
    cache = Path(os.environ.get("HF_DATASETS_CACHE", "/root/data/huggingface/datasets"))
    candidates = sorted(cache.glob("allenai___ai2_arc/ARC-Challenge/*/*"))
    source = next((path for path in candidates if (path / "ai2_arc-test.arrow").is_file()), None)
    if source is None:
        raise FileNotFoundError(f"ARC-Challenge Arrow cache not found below {cache}")
    local = DatasetDict({
        split: Dataset.from_file(str(source / f"ai2_arc-{split}.arrow"))
        for split in ("train", "validation", "test")
    })
    original = datasets.load_dataset

    def load_dataset(*args, **kwargs):
        path = kwargs.get("path", args[0] if args else None)
        name = kwargs.get("name", args[1] if len(args) > 1 else None)
        if path == "allenai/ai2_arc" and name == "ARC-Challenge":
            return local
        return original(*args, **kwargs)

    datasets.load_dataset = load_dataset

    def restore() -> None:
        datasets.load_dataset = original

    return restore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--policy-json', type=Path)
    group.add_argument('--uniform-method', choices=('dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4', 'marlin_nvfp4'))
    parser.add_argument('--label', required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batch-size', default='4')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model-path', type=Path, default=MODEL)
    parser.add_argument('--prepared-root', type=Path, default=PREPARED)
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


def install_prefill_policy(
    model: nn.Module, policy: dict, prepared_root: Path
) -> list[tuple[nn.Module, str, nn.Module]]:
    saved = []
    for method, artifact in STATE.items():
        selected = [
            name for name, entry in policy['method_map'].items()
            if entry['prefill_method'] == method
        ]
        if not selected:
            continue
        state = torch.load(prepared_root / artifact / 'model.pt', map_location='cpu')['state_dict']
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
    if args.policy_json:
        policy = json.loads(args.policy_json.read_text())
        policy_source = str(args.policy_json)
    else:
        method = 'w4a16_ours' if args.uniform_method == 'marlin_nvfp4' else args.uniform_method
        names = [
            f'model.layers.{layer}.{part}.{kind}'
            for layer in range(32)
            for part, kind in (
                ('self_attn', 'qkv_proj'), ('self_attn', 'o_proj'),
                ('mlp', 'gate_up_proj'), ('mlp', 'down_proj'),
            )
        ]
        policy = {'method_map': {name: {'prefill_method': method} for name in names}}
        policy_source = f'uniform:{args.uniform_method}'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='eager',
    ).to(device).eval()
    saved = install_prefill_policy(model, policy, args.prepared_root)
    restore_dataset_loader = None
    try:
        restore_dataset_loader = install_local_arc_dataset()
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(
            pretrained=model, tokenizer=str(args.model_path), backend='causal', dtype=torch.bfloat16,
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
            'policy_json': policy_source,
            'limit': args.limit,
            'batch_size': args.batch_size,
            'num_samples': result.get('n-samples', {}).get('arc_challenge'),
            'acc': metrics.get('acc,none'),
            'acc_norm': metrics.get('acc_norm,none'),
            'raw_metrics': metrics,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        if restore_dataset_loader is not None:
            restore_dataset_loader()
        for obj, child, old in saved:
            setattr(obj, child, old)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
