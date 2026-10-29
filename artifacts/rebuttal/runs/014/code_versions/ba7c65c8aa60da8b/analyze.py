"""Build diagnosis evidence from raw traces and all retained controls."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
POLICIES = ['uniform_dense_bf16', 'uniform_w4a16_ours', 'uniform_marlin_int4']


def read(path):
    return json.loads(path.read_text())


def csv_file(name, rows):
    with (RUN / 'results' / name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    controls = []
    for p in sorted((RUN / 'results/generation_controls').glob('*.json')):
        if p.name.endswith('_trace.json'):
            continue
        d = read(p)
        assert d['complete'] and len(d['samples']) == 11
        timed = [x for x in d['samples'] if not x['warmup']]
        controls.append({'label': p.stem, 'policy': d['policy'], 'mode': d['mode'],
            **d['medians'], 'timed_output_variants': d['timed_output_variants'],
            'median_cpu_mhz_after': statistics.median(x['gpu_after']['host_cpu_khz'] / 1000 for x in timed),
            'median_gpu_mhz_after': statistics.median(x['gpu_after']['sm_mhz'] for x in timed)})
    csv_file('generation_controls.csv', controls)
    fixed, counts = [], []
    for policy in POLICIES:
        d = read(RUN / f'results/{policy}/summary.json')
        assert d['complete'] and d['graph_max_abs_diff'] == 0
        assert d['fixed_inputs']['positions'] == [2110] * 8
        assert d['metadata_fields']['seq_lens'] == [2111] * 8
        events = read(RUN / f'results/{policy}/fixed_eager_trace.json')['traceEvents']
        cpu = Counter(e['name'] for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op')
        kernels = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'kernel']
        categories = Counter()
        for e in kernels:
            n = e['name'].lower()
            category = ('linear' if any(x in n for x in ['marlin', 'gemm', 'matmul']) else
                        'attention' if any(x in n for x in ['flash', 'attention', 'paged']) else 'other')
            categories[category] += e['dur'] / 3000
        fixed.append({'policy': policy, 'eager_ms': d['fixed_eager']['wall_ms'],
            'direct_dispatch_after_profiler_ms': d['fixed_direct_dispatch']['wall_ms'],
            'graph_ms': d['fixed_graph']['wall_ms'], 'graph_max_abs_diff': d['graph_max_abs_diff'],
            'profile_gpu_kernel_sum_ms': sum(e['dur'] for e in kernels) / 3000,
            'profile_linear_ms': categories['linear'], 'profile_attention_ms': categories['attention'],
            'profile_other_ms': categories['other']})
        counts.append({'policy': policy, 'aten_empty_per_forward': cpu['aten::empty'] / 3,
                       'gpu_kernels_per_forward': len(kernels) / 3})
    csv_file('fixed_decode.csv', fixed)
    csv_file('operator_counts.csv', counts)
    paired = []
    for policy in POLICIES:
        path = RUN / f'results/paired/{policy}.json'
        if not path.exists():
            continue
        d = read(path)
        assert d['complete'] and len(d['samples']) == 22
        for mode in ['eager', 'graph']:
            rows = [x for x in d['samples'] if not x['warmup'] and x['mode'] == mode]
            assert len(rows) == 8 and all(x['decode_calls'] == 63 for x in rows)
            paired.append({'policy': policy, 'mode': mode, **d['medians'][mode],
                'min_tpot_ms': min(x['tpot_ms'] for x in rows),
                'max_tpot_ms': max(x['tpot_ms'] for x in rows),
                'output_variants': len({x['output_sha256'] for x in rows})})
    if paired:
        csv_file('paired_generation.csv', paired)
    history = []
    for r in range(3):
        for policy in [*POLICIES, 'uniform_dense_bf16_bracket_end',
                       'hawq_fp4_phase_pair_signed_prefill_decode_q0.07']:
            d = read(RUN.parent / f'013/results/speed/node0/round{r}/{policy}.json')
            rows = [x for c in d['cases'] if c['output'] == 64 for x in c['samples'] if not x['warmup']]
            history.append({'round': r, 'policy': policy,
                'tpot_ms': statistics.median(x['decode_remainder_ms'] / 63 for x in rows),
                'cpu_mhz_after': statistics.median(x['gpu_after']['host_cpu_khz'] / 1000 for x in rows),
                'gpu_mhz_after': statistics.median(x['gpu_after']['sm_mhz'] for x in rows)})
    csv_file('historical_cpu_telemetry.csv', history)
    # Verify the sealed prior experiment rather than modifying it.
    original = RUN.parent / '013'
    manifest = read(original / 'results/manifest.json')['files']
    for name, entry in manifest.items():
        path = original / name
        assert path.stat().st_size == entry['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], name
    formal = [x for x in controls if not x['label'].endswith('_probe')]
    complete = len(formal) == 18 and len(paired) == 6
    audit = {'complete': complete, 'fixed_decode_policies': len(fixed),
             'generation_control_processes': len(formal), 'paired_processes': len(paired) // 2,
             'protected_run013_files_verified': len(manifest),
             'fixed_graph_max_abs_diff': max(x['graph_max_abs_diff'] for x in fixed),
             'scope': 'Uniform methods only; no replacement of mixed DASO/HAWQ main-table timings.'}
    (RUN / 'results/audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
