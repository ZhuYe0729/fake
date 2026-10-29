"""Prespecified central comparisons and the two selected quality budgets."""
from collections import defaultdict

from common import RUN, write_json
from launch import read


def main():
    assert not (RUN / 'configs/full_quality_jobs.json').exists()
    pool = read(RUN / 'configs/allocation_pool.json')
    selected = read(RUN / 'configs/selected_policies.json')
    pairs = []
    for p in pool['points']:
        if not p['policy_id'] or p.get('latency_fraction') != .80:
            continue
        if p['family'] in ['daso matched', 'hawq matched', 'daso independent matched formats']:
            pairs.append({'label': p['label'], 'policy': p['policy_id'], 'scenario': p['scenario']})
    pairs.extend({'label': p['label'], 'policy': p['policy'], 'scenario': p['scenario']} for p in selected['selected'])
    # Fresh full tasks for every selected unique effective map: no mixing old
    # eager and decode-Graph generation when variable-size EOS batches pad.
    jobs = defaultdict(list)
    for p in pairs:
        kind = 'tasks' if p['scenario'] == 'prefill_only' else 'generation'
        if kind not in jobs[p['policy']]:
            jobs[p['policy']].append(kind)
    write_json(RUN / 'configs/full_quality_targets.json', pairs)
    write_json(RUN / 'configs/full_quality_jobs.json', [{'job_id': pid, 'policy': pid, 'kinds': kinds}
        for pid, kinds in jobs.items()])
    print('Fresh full-task policies', len(jobs), 'groups', sum(len(v) for v in jobs.values()), flush=True)


if __name__ == '__main__':
    main()
