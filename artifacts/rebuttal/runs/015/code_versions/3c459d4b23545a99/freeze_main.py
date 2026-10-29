"""Freeze new main rows, inherited quality, and all formal timing orders."""
import hashlib
import json

from common import RUN, write_json


def read(path):
    return json.loads(path.read_text())


def map_key(mapping, scenario):
    value = {n: x['prefill_method'] for n, x in mapping.items()} if scenario == 'prefill_only' else mapping
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    assert not (RUN / 'configs/design.json').exists()
    from prepare_hawq_controls import main as freeze_hawq
    freeze_hawq()
    pool = read(RUN / 'configs/reanalysis_pool.json')
    controls = read(RUN / 'configs/hawq_latency_controls.json')['points']
    selected = read(RUN / 'configs/selected_reanalysis.json')['selected']
    old = read(RUN.parent / '013/configs/design.json')
    lookup = {p['label']: p['policy_id'] for p in pool['points']}
    lookup.update({p['label']: p['policy_id'] for p in controls})
    lookup.update({p['label']: p['policy_id'] for p in selected if p['policy_id']})
    rows = [json.loads(json.dumps(row)) for row in old['rows'][:10]]
    pairs = [
        ('HAWQ-v3 INT4, latency target=.80 BF16 (PTQ)', 'graph_hawq_int4_{s}_r0.8'),
        ('HAWQ-v3 FP4, latency target=.80 BF16 (PTQ)', 'graph_hawq_fp4_{s}_r0.8'),
        ('HAWQ FP4 minimum-cost control, q=.07', 'graph_hawq_fp4_{s}_q0.07'),
        ('DASO, 5 candidates, fresh-profile DP', None),
        ('DASO + 8bit, fresh-profile DP', None),
        ('DASO + 8bit + exact/validation, epsilon=.01', 'graph_verified_{s}_q0.01'),
        ('DASO + 8bit + exact/validation, epsilon=.04', 'graph_verified_{s}_q0.04')]
    for index, (method, pattern) in enumerate(pairs):
        row = {'method': method}
        for scenario in ['prefill_only', 'prefill_decode']:
            if index == 3:
                label = f'graph_dp5_{scenario}_q' + ('0.02' if scenario == 'prefill_only' else '0.07')
            elif index == 4:
                label = f'graph_dp7_{scenario}_q' + ('0.02' if scenario == 'prefill_only' else '0.01')
            else:
                label = pattern.format(s=scenario)
            row[scenario] = {'policy': lookup[label], 'selection_label': label}
        rows.append(row)
    # Preserve existing full evaluations only when the effective weights match.
    sources = {}
    for row in old['rows']:
        for scenario in ['prefill_only', 'prefill_decode']:
            pid = row[scenario]['policy']
            mapping = read(RUN.parent / f'013/configs/policies/{pid}.json')['method_map']
            sources.setdefault((map_key(mapping, scenario), scenario), row[scenario]['quality'])
    quality_jobs = {}
    for row in rows:
        for scenario in ['prefill_only', 'prefill_decode']:
            pid = row[scenario]['policy']
            mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
            source = sources.get((map_key(mapping, scenario), scenario))
            if source is None:
                source = ['015', pid]
                kind = 'tasks' if scenario == 'prefill_only' else 'generation'
                quality_jobs[pid, kind] = {'policy': pid, 'kind': kind}
            row[scenario]['quality'] = source
    policies = {}

    def include(pid, scenario):
        policies.setdefault(pid, {'outputs': []})
        output = 1 if scenario == 'prefill_only' else 64
        if output not in policies[pid]['outputs']:
            policies[pid]['outputs'].append(output)

    for row in [*rows, *old['rows']]:
        for scenario in ['prefill_only', 'prefill_decode']:
            include(row[scenario]['policy'], scenario)
    for point in selected:
        if point['policy_id']:
            include(point['policy_id'], point['scenario'])
    for point in controls:
        include(point['policy_id'], point['scenario'])
    for point in pool['points']:
        if point['family'] == 'fresh HAWQ PTQ adaptation':
            include(point['policy_id'], point['scenario'])
    for p in policies.values():
        p['outputs'].sort()
    other = sorted(set(policies) - {'uniform_dense_bf16'})
    rounds = []
    for number, order in enumerate([other, other[::-1], other[7:] + other[:7]]):
        rounds.append({'round': number, 'order': ['uniform_dense_bf16', *order]})
    write_json(RUN / 'configs/design.json', {'rows': rows, 'fixed_rows': old['rows'],
        'quality_jobs': list(quality_jobs.values()), 'policies': policies, 'rounds': rounds,
        'speed_inputs_sha256': hashlib.sha256((RUN / 'configs/speed_inputs.json').read_bytes()).hexdigest(),
        'seed': 144983, 'warmups': 3, 'iterations': 8, 'kv_bytes': 6 * 1024**3,
        'runtime': 'Phase-reused fixed addresses, device-alpha FP4 bindings, native FULL_DECODE_ONLY graph; asynchronous H2D retained.',
        'selection_sha256': hashlib.sha256((RUN / 'configs/selected_reanalysis.json').read_bytes()).hexdigest(),
        'scope': 'Main 17 rows reoptimized; fixed old 17 rows retained as runtime ablation. Three quiet same-card formal campaigns.'})
    print('Main rows', len(rows), 'new full-quality jobs', len(quality_jobs),
          'formal policies', len(policies), 'processes', 3 * (len(policies) + 1), flush=True)


if __name__ == '__main__':
    main()
