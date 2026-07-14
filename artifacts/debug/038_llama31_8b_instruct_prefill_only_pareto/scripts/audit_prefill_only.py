#!/usr/bin/env python3
"""Freeze the Llama3.1 prefill-only architecture, actions, and runner protocol."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/home/agent/wja/project/my/cospaq/fake')
MODEL = Path('/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct')
CUTLASS = REPO / 'fake/kernels/cutlass/cutlass_wrapper'
sys.path[:0] = [str(REPO), str(CUTLASS), str(CUTLASS / 'modeling')]

from transformers import AutoConfig  # noqa: E402
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402

KERNELS = ('dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4', 'marlin_nvfp4')
CONVERSIONS = {'dense_nvfp4': 'canonical_to_cutlass', 'marlin_nvfp4': 'canonical_to_marlin'}


def linears(config):
    hidden, intermediate = int(config.hidden_size), int(config.intermediate_size)
    head_dim = int(getattr(config, 'head_dim', 0) or hidden // int(config.num_attention_heads))
    kv_heads = int(getattr(config, 'num_key_value_heads', config.num_attention_heads))
    qkv = hidden + 2 * kv_heads * head_dim
    shapes = [('qkv_proj', qkv, hidden), ('o_proj', hidden, hidden),
              ('gate_up_proj', 2 * intermediate, hidden), ('down_proj', hidden, intermediate)]
    return [(f'model.layers.{layer}.{group}.{name}', layer, name, n, k)
            for layer in range(int(config.num_hidden_layers))
            for group, name, n, k in [('self_attn', *shapes[0]), ('self_attn', *shapes[1]),
                                      ('mlp', *shapes[2]), ('mlp', *shapes[3])]]


def candidate_row(candidate):
    return {'supported': bool(candidate.supported), 'latency_ms': candidate.latency_ms,
            'source': candidate.source, 'reason': candidate.reason,
            'prediction_status': candidate.prediction_status,
            'prediction_error': candidate.prediction_error}


def main():
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    specs = linears(config)
    predictor = KernelLatencyPredictor(model_root=DEFAULT_MODEL_ROOT, kernels=KERNELS)
    rows = []
    for fused, layer, module_type, n, k in specs:
        query = predictor.predict(8 * 2048, n, k)
        by_kernel = {x.kernel: x for x in query.candidates}
        conversion = {x.conversion: x for x in predictor.predict_conversion(n, k)}
        for kernel in KERNELS:
            row = {'module_name': fused, 'layer': layer, 'module_type': module_type,
                   'm': 8 * 2048, 'n': n, 'k': k, 'kernel': kernel,
                   **candidate_row(by_kernel[kernel])}
            if kernel in CONVERSIONS:
                conv = conversion.get(CONVERSIONS[kernel])
                row.update({'conversion': CONVERSIONS[kernel],
                            'conversion_supported': bool(conv and conv.supported),
                            'conversion_ms': None if conv is None else conv.latency_ms,
                            'conversion_reason': '' if conv is None else conv.reason})
            else:
                row.update({'conversion': '', 'conversion_supported': True,
                            'conversion_ms': 0.0, 'conversion_reason': ''})
            rows.append(row)
    protocol = {'scenario': 'prefill_only', 'batch': 8, 'input_tokens': 2048,
                'api_output_tokens': 1, 'm_prefill': 16384,
                'runner': {'enforce_eager': True, 'enable_prefix_caching': False,
                           'enable_chunked_prefill': False, 'gpu_memory_utilization': 0.9,
                           'fresh_process_repetitions': 5, 'timing_scope': 'generate_only_after_loaded_llm'},
                'methods': list(KERNELS), 'logical_runtime_mapping': {
                    'marlin_nvfp4': 'w4a16_ours'},
                'model_path': str(MODEL), 'module_count': len(specs),
                'config': {key: int(getattr(config, key)) for key in ('num_hidden_layers', 'hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads')},
                'head_dim': int(getattr(config, 'head_dim', 0) or config.hidden_size // config.num_attention_heads),
                'fused_shapes': {name: [n, k] for name, _, _, n, k in specs[:4]}}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'architecture_manifest.json').write_text(json.dumps(protocol, indent=2) + '\n')
    with (ROOT / 'action_support.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    bad = [r for r in rows if not r['supported'] or not r['conversion_supported']]
    print(json.dumps({'manifest': str(ROOT / 'architecture_manifest.json'), 'actions': len(rows),
                      'unsupported_actions': len(bad), 'module_count': len(specs)}, indent=2))


if __name__ == '__main__':
    main()
