"""Held-out quality and frozen actual-latency limits, with no reselection."""
import csv
import json
import statistics

import numpy as np

from common import RUN, write_json


def read(path):
    return json.loads(path.read_text())


def save_csv(name, rows):
    with (RUN / 'results' / name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quality(pid, scenario):
    result = read(RUN / f'results/test/{pid}/{scenario}.json')
    baseline = read(RUN / f'results/test/uniform_dense_bf16/{scenario}.json')
    differences = np.array([a['avg_nll'] - b['avg_nll'] for a, b in zip(result['blocks'], baseline['blocks'], strict=True)])
    assert len(differences) == 64
    return float(differences.mean()), float(differences.std(ddof=1) / 8)


def main():
    selected = read(RUN / 'configs/selected_reanalysis.json')['selected']
    rows = []
    for point in selected:
        pid, scenario = point['policy_id'], point['scenario']
        row = {'label': point['label'], 'policy': pid, 'scenario': scenario, 'family': point['family'],
               'quality_budget': point.get('quality_budget'), 'frozen_e2e_limit_ms': point.get('actual_e2e_limit_ms'),
               'test_delta_nll': None, 'paired_se': None, 'test_mean_quality_violation': None,
               'median_e2e_ms': None, 'median_cycle_ms': None, 'generations': 0,
               'e2e_limit_violations': None, 'cycle_limit_violations': None}
        if pid:
            mean, se = quality(pid, scenario)
            output = 1 if scenario == 'prefill_only' else 64
            observations = []
            for number in range(3):
                data = read(RUN / f'results/speed/node0/round{number}/{pid}.json')
                case = next(c for c in data['cases'] if c['output'] == output)
                observations.extend(s for s in case['samples'] if not s['warmup'])
            assert len(observations) == 24
            row.update(test_delta_nll=mean, paired_se=se, generations=len(observations),
                       median_e2e_ms=statistics.median(s['e2e_ms'] for s in observations),
                       median_cycle_ms=statistics.median(s['cycle_ms'] for s in observations))
            if 'quality_budget' in point:
                row['test_mean_quality_violation'] = mean > point['quality_budget'] + 1e-12
            if 'actual_e2e_limit_ms' in point:
                cap = point['actual_e2e_limit_ms']
                row['e2e_limit_violations'] = sum(s['e2e_ms'] > cap for s in observations)
                row['cycle_limit_violations'] = sum(s['cycle_ms'] > cap for s in observations)
        rows.append(row)
    save_csv('selected_test_constraints.csv', rows)
    raw = []
    for point in read(RUN / 'configs/reanalysis_pool.json')['points']:
        if 'quality_budget' not in point or not point['family'].startswith('fresh '):
            continue
        pid, scenario = point['policy_id'], point['scenario']
        path = RUN / f'results/test/{pid}/{scenario}.json'
        if not path.exists():
            continue
        mean, se = quality(pid, scenario)
        raw.append({'label': point['label'], 'policy': pid, 'scenario': scenario,
            'family': point['family'], 'quality_budget': point['quality_budget'],
            'predicted_delta_nll': point['predicted_delta_nll'], 'test_delta_nll': mean,
            'paired_se': se, 'test_mean_quality_violation': mean > point['quality_budget'] + 1e-12})
    save_csv('raw_proxy_test_constraints.csv', raw)
    quality_rows = [r for r in rows if r['quality_budget'] is not None]
    inverse = [r for r in rows if r['family'] == 'actual inverse validation-selected']
    write_json(RUN / 'results/constraints_summary.json', {
        'quality_budget_points': len(quality_rows),
        'quality_unique_policy_scenarios': len({(r['policy'], r['scenario']) for r in quality_rows}),
        'heldout_mean_quality_violations': sum(r['test_mean_quality_violation'] for r in quality_rows),
        'inverse_points': len(inverse), 'inverse_feasible': sum(bool(r['policy']) for r in inverse),
        'inverse_unique_policy_scenarios': len({(r['policy'], r['scenario']) for r in inverse if r['policy']}),
        'inverse_e2e_violations': sum(r['e2e_limit_violations'] or 0 for r in inverse),
        'inverse_cycle_violations': sum(r['cycle_limit_violations'] or 0 for r in inverse),
        'scope': 'Empirical means on 64 held-out windows; actual E2E thresholds frozen from calibration. Cycle includes restore and is separately audited. Reused maps are not independent policy experiments.'})
    print(read(RUN / 'results/constraints_summary.json'), flush=True)


if __name__ == '__main__':
    main()
