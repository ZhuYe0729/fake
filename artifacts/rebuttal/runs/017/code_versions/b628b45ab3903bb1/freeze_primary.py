"""Prioritize both models' core main tables, without dropping supplemental points."""
from collections import defaultdict

from common import RUN, write_json
from launch import read


def main():
    assert not list((RUN / 'results/confirmation').glob('*/*.json'))
    pool = read(RUN / 'configs/allocation_pool.json')
    selected = read(RUN / 'configs/selected_policies.json')
    targets = []
    primary = set()
    for point in pool['points']:
        if not point['policy_id']:
            continue
        core = point['family'] in ['uniform anchor', 'original DASO-5']
        hawq = point['family'] == 'hawq matched' and point.get('latency_fraction') == .80 and (
            point['scenario'] == 'prefill_only' or point.get('phase_mode') == 'phase_pair')
        simple_phase = point['label'] == 'phase_sq_int8_w4a16_ours'
        central = point['family'] in ['daso matched', 'hawq matched'] and point.get('latency_fraction') == .80
        if core or central or simple_phase:
            primary.add(point['policy_id'])
        if core or central or simple_phase:
            targets.append({'label':point['label'],'policy':point['policy_id'], 'scenario':point['scenario']})
    for point in selected['selected']:
        primary.add(point['policy'])
        targets.append({'label':point['label'],'policy':point['policy'],'scenario':point['scenario']})
    write_json(RUN / 'configs/primary_policies.json', {'policies':sorted(primary),
        'scope':'Prespecified core uniform/original main rows, central same-cap controls and validation-selected DASO-8; no confirmation read. Other points retained for supplement after both model core tables.'})
    jobs = defaultdict(list)
    for point in targets:
        kind = 'tasks' if point['scenario'] == 'prefill_only' else 'generation'
        if kind not in jobs[point['policy']]:
            jobs[point['policy']].append(kind)
    write_json(RUN / 'configs/full_quality_targets.json', targets)
    write_json(RUN / 'configs/full_quality_jobs.json', [{'job_id':pid, 'policy':pid, 'kinds':kinds}
        for pid,kinds in jobs.items()])
    print('Frozen primary speed policies', len(primary), 'full-task groups', sum(map(len,jobs.values())), flush=True)


if __name__ == '__main__':
    main()
