"""Read-only diagnostics of the sealed run016, without fitting on confirmation."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / 'runs/016'
SOURCES = {}


def read(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(name, rows):
    with (ROOT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(rid):
    assert read(RUN / 'results/formal_complete.json')['complete']
    pool = read(RUN / 'configs/allocation_pool.json')
    design = read(RUN / 'configs/frozen_design.json')
    profile_path = RUN / 'reference/graph_action_profile.csv'
    SOURCES[str(profile_path)] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    lut = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
           for r in csv.DictReader(profile_path.open())}
    calibration = read(RUN / 'reference/calibration_uniform_dense_bf16.json')
    rows, memory = [], []
    for pid, spec in pool['policies'].items():
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        for scene in spec['scenarios']:
            output = 1 if scene == 'prefill_only' else 64
            costs = lambda m: sum(lut[n.rsplit('.', 1)[-1], v['prefill_method'], 16384] +
                (63 * lut[n.rsplit('.', 1)[-1], v['decode_method'], 8] if output == 64 else 0.)
                for n, v in m.items())
            baseline_map = {n: {'prefill_method': 'dense_bf16', 'decode_method': 'dense_bf16'} for n in mapping}
            base = next(c['medians']['e2e_ms'] for c in calibration['cases'] if c['output'] == output)
            offset = base - costs(baseline_map)
            predicted = costs(mapping) + offset
            measured = []
            timed = pid in read(RUN / 'configs/primary_policies.json')['policies']
            for campaign in range(3 if timed else 0):
                raw = read(RUN / f'results/speed/node0/round{campaign}/{pid}.json')
                samples = [s for c in raw['cases'] if c['output'] == output for s in c['samples'] if not s['warmup']]
                measured.append(statistics.median(s['e2e_ms'] for s in samples))
                memory.append({'policy': pid, 'scenario': scene, 'campaign': campaign,
                    'initialization_ms': raw['initialization_ms'], 'capture_model_ms': raw['capture_model_ms'],
                    'restore_ms': statistics.median(s['restore_ms'] for s in samples),
                    'cycle_ms': statistics.median(s['cycle_ms'] for s in samples), **raw['memory']})
            actual = statistics.median(measured) if timed else None
            actions = read(RUN / f'data/actions_{scene}.json')
            predicted_nll = actions['bias'] + sum(actions['quality'][i + j*len(design['names'])][design['methods'].index(mapping[n][phase+'_method'])]
                for j, phase in enumerate(actions['phases']) for i, n in enumerate(design['names']))
            test = read(RUN / f'results/confirmation/{pid}/{scene}.json')
            bf16 = read(RUN / f'results/confirmation/uniform_dense_bf16/{scene}.json')
            delta = statistics.mean(a['avg_nll']-b['avg_nll'] for a,b in zip(test['blocks'],bf16['blocks'],strict=True))
            rows.append({'policy': pid, 'scenario': scene, 'predicted_linear_ms': costs(mapping),
                'fixed_bf16_offset_ms': offset, 'predicted_e2e_ms': predicted, 'measured_e2e_ms': actual,
                'absolute_percentage_error': abs(predicted-actual)/actual*100 if timed else None,
                'signed_percentage_error': (predicted-actual)/actual*100 if timed else None,
                'predicted_delta_nll': predicted_nll, 'confirmation_delta_nll': delta,
                'quality_absolute_error': abs(predicted_nll-delta),
                'families': ';'.join(sorted({p['family'] for p in pool['points'] if p['policy_id']==pid and p['scenario']==scene}))})
    assert len(rows) == sum(len(s['scenarios']) for s in pool['policies'].values())
    save(f'llama{rid}_modeling_points.csv', rows)
    save(f'llama{rid}_runtime_overhead.csv', memory)
    summary = []
    for scene in ['prefill_only', 'prefill_decode']:
        local = [r for r in rows if r['scenario']==scene]
        timed_rows=[r for r in local if r['measured_e2e_ms'] is not None]
        summary.append({'scenario': scene, 'quality_unique_points': len(local), 'latency_timed_points':len(timed_rows),
            'latency_mape_percent': statistics.mean(r['absolute_percentage_error'] for r in timed_rows),
            'latency_median_ape_percent': statistics.median(r['absolute_percentage_error'] for r in timed_rows),
            'latency_max_ape_percent': max(r['absolute_percentage_error'] for r in timed_rows),
            'quality_mae_nats_per_token': statistics.mean(r['quality_absolute_error'] for r in local)})
    save(f'llama{rid}_modeling_summary.csv', summary)
    (ROOT / f'llama{rid}_modeling_scope.json').write_text(json.dumps({
        'scope': 'Quality covers every frozen policy/scenario; latency covers exactly the prespecified core primary timings, with supplements explicitly pending. No confirmation refitting. '
                 'Actual measured-LUT Linear sum plus previously fixed BF16 E2E offset; not the original broad Hybrid predictor. '
                 'NLL predictions use frozen eight-action quality model, including poor raw proposals. '
                 'Memory is process-global peak across both scenarios, not isolated per-case memory. '
                 'Per-process durations, host caches, restore, capture and cycle are not interchangeable.',
        'sources_sha256': SOURCES}, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    for rid in ['017','018']:
        RUN=ROOT.parent/'runs'/rid
        if (RUN/'results/formal_complete.json').exists():
            SOURCES.clear()
            analyze(rid)
