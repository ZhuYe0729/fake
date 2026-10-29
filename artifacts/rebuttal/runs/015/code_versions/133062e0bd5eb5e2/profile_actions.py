"""Fresh matched costs: eager prefill and cold-working-set decode graphs."""
import copy
import csv
import gc
import json
import math
import os
import statistics
import time

import torch

from common import RUN, compressed_module, write_json
from eightbit import EightBitLinear

PARTS = {'qkv_proj': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
         'o_proj': ['self_attn.o_proj'], 'gate_up_proj': ['mlp.gate_proj', 'mlp.up_proj'],
         'down_proj': ['mlp.down_proj']}
METHODS = ['dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4',
           'w4a16_ours', 'int8', 'marlin_int4', 'sq_int8', 'fp8', 'sq_fp8']


@torch.inference_mode()
def main():
    from graph_bindings import install
    install()
    target = RUN / 'results/graph_action_profile.csv'
    assert not target.exists(), 'Preserve the completed profile.'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    weights = torch.load(RUN / 'data/profile_weights.pt', mmap=True, weights_only=True)
    scales = torch.load(RUN / 'data/activation_scales.pt', weights_only=True)
    states = {m: torch.load('/dev/shm/daso_rebuttal_001_20261025/canonical/' + m + '/model.pt',
                           mmap=True, weights_only=True)['state_dict'] for m in ['sparse_bf16', 'sparse_nvfp4']}
    torch.manual_seed(148015)
    rows = []
    started = time.perf_counter()
    for name, raw in weights.items():
        kind = name.rsplit('.', 1)[-1]
        for method in METHODS:
            weight = raw.cuda()
            if method in states:
                weight = torch.cat([states[method]['model.layers.0.' + p + '.weight']
                                    for p in PARTS[kind]]).cuda()
            layer = (EightBitLinear(method, name, weight, scales)
                     if method in ['sq_int8', 'fp8', 'sq_fp8'] else
                     compressed_module(method, weight, int8_backend='int_mm')).eval()
            for m in [8, 16384]:
                x = torch.randn(m, weight.shape[1], dtype=torch.bfloat16, device='cuda')
                copies = [layer]
                graph = None
                # Full-model decoding streams weights from many different layers.
                # Avoid measuring one tiny weight matrix repeatedly from L2.
                if m == 8:
                    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                                for t in [*layer.parameters(), *layer.buffers()]}
                    count = min(64, max(4, math.ceil(512 * 1024**2 / sum(storages.values()))))
                    copies += [copy.deepcopy(layer) for _ in range(count - 1)]
                    for current in copies:
                        out = current(x)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for current in copies:
                            out = current(x)
                    call, divisor, repeats = graph.replay, count, 5
                else:
                    call, divisor, repeats = lambda: layer(x), 1, 3
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                samples = []
                for _ in range(5):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    start = time.perf_counter()
                    for _ in range(repeats):
                        call()
                    end.record()
                    end.synchronize()
                    samples.append({'wall_ms': (time.perf_counter() - start) * 1000 / repeats / divisor,
                                    'stream_ms': begin.elapsed_time(end) / repeats / divisor})
                rows.append({'module_type': kind, 'method': method, 'm': m,
                    'n': weight.shape[0], 'k': weight.shape[1],
                    'latency_ms': statistics.median(s['stream_ms'] if m == 8 else s['wall_ms'] for s in samples),
                    'protocol': 'decode_graph_cold_working_set' if m == 8 else 'prefill_eager_forward',
                    'distinct_weight_copies': len(copies),
                    'samples': json.dumps(samples)})
                del call, graph, copies, x
                if m == 8:
                    del current, out
                gc.collect()
                torch.cuda.empty_cache()
            print(kind, method, 'done', round(time.perf_counter() - started, 1), flush=True)
            write_json(RUN / 'results/profile_progress.json', {'rows': rows, 'elapsed_seconds': time.perf_counter() - started})
            del layer, weight
            gc.collect()
            torch.cuda.empty_cache()
    with target.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(RUN / 'results/profile_complete.json', {'complete': True, 'rows': len(rows),
        'seconds': time.perf_counter() - started, 'gpu': 0, 'methods': METHODS,
        'scope': 'Matched Linear forwards; additive proxy only, excludes attention/scheduling/H2D overlap.'})


if __name__ == '__main__':
    main()
