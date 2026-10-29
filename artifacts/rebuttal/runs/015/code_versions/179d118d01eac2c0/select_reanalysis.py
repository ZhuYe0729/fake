"""Freeze choices using validation quality and calibration timing only."""
import hashlib
import json

import numpy as np

from common import RUN, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    assert not (RUN / 'configs/selected_reanalysis.json').exists()
    assert not list((RUN / 'results/test').glob('*/*.json'))
    pool = read(RUN / 'configs/reanalysis_pool.json')
    rules = read(RUN / 'configs/selection_rules.json')
    assert hashlib.sha256((RUN / 'configs/reanalysis_pool.json').read_bytes()).hexdigest() == rules['pool_sha256']
    values = []
    for scenario in ['prefill_only', 'prefill_decode']:
        reference = read(RUN / f'results/validation/uniform_dense_bf16/{scenario}.json')
        output = 1 if scenario == 'prefill_only' else 64
        baseline = next(c for c in read(RUN / 'results/measurements/calibration_uniform_dense_bf16.json')['cases'] if c['output'] == output)
        for pid, p in pool['policies'].items():
            if not p['daso_eligible'] or scenario not in p['scenarios']:
                continue
            result = read(RUN / f'results/validation/{pid}/{scenario}.json')
            speed = read(RUN / f'results/measurements/calibration_{pid}.json')
            case = next(c for c in speed['cases'] if c['output'] == output)
            differences = np.array([a['avg_nll'] - b['avg_nll'] for a, b in zip(result['blocks'], reference['blocks'], strict=True)])
            mean, se = float(differences.mean()), float(differences.std(ddof=1) / np.sqrt(64))
            guard = 0. if pid == 'uniform_dense_bf16' else mean + 2 * se + .005
            maximum = max(x['e2e_ms'] for x in case['samples'] if not x['warmup'])
            values.append({'policy_id': pid, 'scenario': scenario, 'delta_nll': mean,
                'paired_se': se, 'guarded_delta_nll': guard,
                'calibration_median_ms': case['medians']['e2e_ms'],
                'calibration_max_ms': maximum, 'guarded_calibration_ms': maximum * 1.05,
                'calibration_cycle_ms': case['medians']['cycle_ms'],
                'baseline_calibration_ms': baseline['medians']['e2e_ms']})
    chosen = []
    for scenario in ['prefill_only', 'prefill_decode']:
        local = [x for x in values if x['scenario'] == scenario]
        for budget in rules['quality_budgets']:
            feasible = [x for x in local if x['guarded_delta_nll'] <= budget]
            winner = min(feasible, key=lambda x: (x['calibration_median_ms'], x['policy_id']))
            chosen.append({'label': f'graph_verified_{scenario}_q{budget:g}', 'scenario': scenario,
                'policy_id': winner['policy_id'], 'quality_budget': budget,
                'family': 'validation-selected', 'validation': winner, 'feasible_candidates': len(feasible)})
        for fraction in rules['inverse_latency_fractions']:
            cap = local[0]['baseline_calibration_ms'] * fraction
            feasible = [x for x in local if x['guarded_calibration_ms'] <= cap]
            winner = min(feasible, key=lambda x: (x['guarded_delta_nll'], x['calibration_median_ms'])) if feasible else None
            chosen.append({'label': f'graph_verified_inverse_{scenario}_r{fraction:g}', 'scenario': scenario,
                'policy_id': winner['policy_id'] if winner else None, 'latency_fraction': fraction,
                'actual_e2e_limit_ms': cap, 'family': 'actual inverse validation-selected',
                'validation': winner, 'feasible_candidates': len(feasible)})
    write_json(RUN / 'results/validation_summary.json', values)
    write_json(RUN / 'configs/selected_reanalysis.json', {'selected': chosen,
        'rules_sha256': hashlib.sha256((RUN / 'configs/selection_rules.json').read_bytes()).hexdigest(),
        'pool_sha256': rules['pool_sha256'], 'selection_frozen_before_new_test_labels': True})
    for point in chosen:
        print(point['label'], point['policy_id'], flush=True)


if __name__ == '__main__':
    main()
