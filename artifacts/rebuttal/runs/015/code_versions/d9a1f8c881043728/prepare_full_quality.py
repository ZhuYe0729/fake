"""Freeze required test labels and downstream jobs after validation selection."""
import hashlib
import json

from common import RUN, write_json
from prepare_quality import signature


def read(path):
    return json.loads(path.read_text())


def main():
    assert (RUN / 'configs/selected_reanalysis.json').exists()
    assert not (RUN / 'configs/full_quality_jobs.json').exists()
    design = read(RUN / 'configs/design.json')
    pool = read(RUN / 'configs/reanalysis_pool.json')
    selected = read(RUN / 'configs/selected_reanalysis.json')['selected']
    pairs = set()
    for row in design['rows']:
        for scenario in ['prefill_only', 'prefill_decode']:
            pairs.add((row[scenario]['policy'], scenario))
    for p in pool['points']:
        if p['family'] in ['fresh exact surrogate', 'fresh inverse surrogate',
                          'fresh original DP 7 actions', 'fresh HAWQ PTQ adaptation']:
            pairs.add((p['policy_id'], p['scenario']))
    for p in selected:
        if p['policy_id']:
            pairs.add((p['policy_id'], p['scenario']))
    for p in read(RUN / 'configs/hawq_latency_controls.json')['points']:
        pairs.add((p['policy_id'], p['scenario']))
    # Freeze the test request before opening any prior test result for inheritance.
    write_json(RUN / 'configs/test_requests.json', {'pairs': sorted(pairs),
        'selection_sha256': hashlib.sha256((RUN / 'configs/selected_reanalysis.json').read_bytes()).hexdigest()})
    dataset_sha = hashlib.sha256((RUN / 'data/test_64_2112.pt').read_bytes()).hexdigest()
    cache = {}
    for rid in ['005', '009', '012']:
        previous = RUN.parent / rid
        for source in sorted((previous / 'results/test').glob('*/*.json')):
            data = read(source)
            if data.get('data_sha256') != dataset_sha or len(data.get('blocks', [])) != 64:
                continue
            policy = previous / f'configs/policies/{source.parent.name}.json'
            if policy.exists():
                cache.setdefault((signature(read(policy)['method_map'], source.stem), source.stem), source)
    jobs = {}
    inherited = []
    for pid, scenario in sorted(pairs):
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        source = cache.get((signature(mapping, scenario), scenario))
        if source:
            data = read(source)
            data['inherited_source'] = str(source)
            data['inheritance_scope'] = 'Identical effective map, canonical weights, frozen data, batch8; numerical binding changes audited.'
            write_json(RUN / f'results/test/{pid}/{scenario}.json', data)
            inherited.append({'policy': pid, 'scenario': scenario, 'source': str(source)})
        else:
            jobs.setdefault(pid, {'policy': pid, 'test_scenarios': [], 'kinds': []})['test_scenarios'].append(scenario)
    for item in design['quality_jobs']:
        job = jobs.setdefault(item['policy'], {'policy': item['policy'], 'test_scenarios': [], 'kinds': []})
        if item['kind'] not in job['kinds']:
            job['kinds'].append(item['kind'])
    ordered = sorted(jobs.values(), key=lambda j: (
        -900 * ('tasks' in j['kinds']) - 900 * ('generation' in j['kinds']) - 45 * len(j['test_scenarios']), j['policy']))
    groups, loads = [[] for _ in range(4)], [0] * 4
    for job in ordered:
        index = min(range(4), key=lambda i: loads[i])
        groups[index].append(job)
        loads[index] += 900 * ('tasks' in job['kinds']) + 900 * ('generation' in job['kinds']) + 45 * len(job['test_scenarios'])
    write_json(RUN / 'configs/full_quality_jobs.json', {'groups': groups, 'jobs': ordered,
        'inherited_test': inherited, 'balance_estimates': loads,
        'scope': 'New main-map full tasks and preregistered raw/selected test NLL; no test-based reselection.'})
    print('Full-quality policies', len(jobs), 'new main task groups', len(design['quality_jobs']),
          'test cases inherited', len(inherited), 'new', sum(len(j['test_scenarios']) for j in ordered), flush=True)


if __name__ == '__main__':
    main()
