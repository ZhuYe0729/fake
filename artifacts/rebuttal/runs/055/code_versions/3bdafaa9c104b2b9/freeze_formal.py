"""Freeze the three-family comparison and deduplicate identical precision maps."""
import hashlib
import json

from common import RUN, write_json


def main():
    target = RUN / 'configs/formal_points.json'
    assert not target.exists()
    design = json.loads((RUN / 'configs/design.json').read_text())
    selected = json.loads((RUN / 'configs/daso_selected.json').read_text())
    hawq = json.loads((RUN / 'results/hawq_allocation.json').read_text())
    points = [{'family': 'BF16', 'policy_id': 'uniform_dense_bf16', 'scenario': scenario}
              for scenario in ['prefill_only', 'prefill_decode']]
    points += [{'family': 'DASO', **point} for point in selected['points']]
    points += [{'family': 'HAWQ-V3 adaptation', **point} for point in hawq['points']]
    cases = {}
    for index, point in enumerate(points):
        policy = point['policy_id']
        path = RUN / f'configs/policies/{policy}.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == design['policies'][policy]['map_sha256']
        mapping = json.loads(path.read_text())['method_map']
        assert len(mapping) == 256
        case = next(case for case in design['workloads'] if case['scenario'] == point['scenario'])
        validation = RUN / f"results/quality/validation/{policy}/{case['label']}.json"
        body = json.loads(validation.read_text())
        assert body['complete'] and len(body['blocks']) == 100
        assert body['sample_sha256'] == design['quality_sha256']['validation']
        signature = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        key = point['scenario'] + '_' + signature[:16]
        point.update(case_key=key, map_signature=signature, validation_result=str(validation))
        if key not in cases:
            cases[key] = {'case_key': key, 'policy_id': policy, 'scenario': point['scenario'],
                          'map_signature': signature, 'aliases': []}
        cases[key]['aliases'].append(index)
    jobs = []
    for case in cases.values():
        if case['scenario'] == 'prefill_only':
            for task in ['wikitext', 'arc_easy', 'arc_challenge', 'winogrande', 'mmlu']:
                jobs.append({'id': case['case_key'] + '_' + task, **case, 'task': task, 'kind': 'multiple_choice', 'begin': 0, 'end': None})
        else:
            for task, count in [('cnn_dm_1000', 1000), ('dsum', 1500), ('IWSLT', 333)]:
                for begin in range(0, count, 512):
                    end = min(begin + 512, count)
                    jobs.append({'id': case['case_key'] + f'_{task}_{begin:04d}_{end:04d}',
                                 **case, 'task': task, 'kind': 'generation', 'begin': begin, 'end': end})
    write_json(target, {'points': points, 'unique_cases': list(cases.values()), 'downstream_jobs': jobs,
        'formal_methods': ['BF16', 'DASO', 'HAWQ-V3 adaptation'], 'speed_rounds': 3,
        'warmups': 5, 'iterations': 10, 'speed_gpu': 0, 'speed_cpu': '0-7', 'speed_numa': 0,
        'selection_rules_sha256': hashlib.sha256((RUN / 'configs/selection_rules.json').read_bytes()).hexdigest(),
        'scope': 'All maps frozen before formal test NLL, timing and downstream scores. Identical maps share evidence; '
                 'shared rows are not independent repeats. HAWQ matches a proxy latency cap, not a true NLL budget.'})
    write_json(RUN / 'results/formal_freeze_complete.json', {'complete': True, 'points': len(points),
        'unique_cases': len(cases), 'downstream_jobs': len(jobs),
        'formal_points_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
