from __future__ import annotations

import csv
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

from common import MODEL, RUN, compressed_module, quantize_int8, write_json


def read_tensor(name, mapping):
    with safe_open(MODEL / mapping[name], framework='pt', device='cpu') as reader:
        return reader.get_tensor(name)


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(0)
    started = time.perf_counter()
    mapping = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    specs = {'self_attn.qkv_proj': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
             'self_attn.o_proj': ['self_attn.o_proj'],
             'mlp.gate_up_proj': ['mlp.gate_proj', 'mlp.up_proj'],
             'mlp.down_proj': ['mlp.down_proj']}
    states = {method: torch.load(Path('/dev/shm/daso_rebuttal_001_20261025/canonical') / method / 'model.pt',
                map_location='cpu', mmap=True, weights_only=True)['state_dict']
              for method in ['sparse_bf16', 'sparse_nvfp4']}
    rows = []
    for suffix, parts in specs.items():
        for method in ['dense_bf16', 'int8', 'int4', 'dense_nvfp4', 'sparse_bf16',
                       'sparse_nvfp4', 'w4a16_ours']:
            names = [f'model.layers.0.{part}.weight' for part in parts]
            source = [states[method][name] if method in states else read_tensor(name, mapping)
                      for name in names]
            weight = torch.cat(source).to('cuda', dtype=torch.bfloat16)
            for m in [1, 8, 2048, 16384]:
                row = {'module_type': suffix, 'method': method, 'm': m,
                       'n': weight.shape[0], 'k': weight.shape[1]}
                try:
                    layer = compressed_module(method, weight, int8_backend='int_mm').eval()
                    x = torch.randn(m, weight.shape[1], dtype=torch.bfloat16, device='cuda')
                    for _ in range(3):
                        layer(x)
                    values = []
                    for _ in range(10):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        output = layer(x)
                        end.record()
                        end.synchronize()
                        values.append(begin.elapsed_time(end))
                    if not torch.isfinite(output).all():
                        raise RuntimeError('non-finite kernel output')
                    row.update(supported=True, latency_ms=statistics.median(values),
                               samples_ms=json.dumps(values))
                    del layer, x, output
                except Exception as exc:
                    row.update(supported=False, error=repr(exc))
                rows.append(row)
                print(row, flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            del weight
    with (RUN / 'results/action_profile.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['module_type','method','m','n','k',
            'supported','latency_ms','samples_ms','error'])
        writer.writeheader()
        writer.writerows(rows)
    errors = []
    from standalone_w4a4_int4.w4a4_int4.linear import _unpack_signed_int4_row_major
    from cutlass_wrapper import quantize_w4a4_int4_weight
    for name in mapping:
        if not name.startswith('model.layers.') or not any(name.endswith(part + '.weight')
                for parts in specs.values() for part in parts):
            continue
        weight = read_tensor(name, mapping).to('cuda', dtype=torch.bfloat16)
        for method in ['int4', 'int8']:
            if method == 'int8':
                packed, scale = quantize_int8(weight)
                dequantized = packed.float() * scale
            else:
                packed = quantize_w4a4_int4_weight(weight)
                dequantized = (_unpack_signed_int4_row_major(packed.packed_weight, weight.shape[1]).float()
                               * packed.weight_scale.reshape(-1, 1))
            errors.append({'module': name.removesuffix('.weight'), 'method': method,
                           'numel': weight.numel(),
                           'weight_error_squared': float((weight.float() - dequantized).square().sum())})
        del weight, packed, dequantized
    with (RUN / 'results/weight_quantization_errors.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0]))
        writer.writeheader()
        writer.writerows(errors)
    write_json(RUN / 'results/action_profile_metadata.json', {
        'gpu': torch.cuda.get_device_name(), 'elapsed_seconds': time.perf_counter() - started,
        'latency_scope': 'complete packaged Linear.forward including activation quantization and scaling',
        'source_weights': 'Qwen3 layer 0 plus canonical SparseGPT states',
        'rows': len(rows), 'unsupported': sum(not row['supported'] for row in rows)})


if __name__ == '__main__':
    main()
