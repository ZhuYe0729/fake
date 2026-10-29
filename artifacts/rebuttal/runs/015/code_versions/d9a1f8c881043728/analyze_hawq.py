"""Report every baseline budget, including infeasible and duplicate points."""
import csv
import statistics

from common import RUN, write_json
from analyze_constraints import quality, read


def main():
    original = [p for p in read(RUN / 'configs/reanalysis_pool.json')['points']
                if p['family'] == 'fresh HAWQ PTQ adaptation']
    controls = read(RUN / 'configs/hawq_latency_controls.json')['points']
    rows = []
    for point in [*original, *controls]:
        pid, scenario = point['policy_id'], point['scenario']
        output = 1 if scenario == 'prefill_only' else 64
        samples = []
        for number in range(3):
            data = read(RUN / f'results/speed/node0/round{number}/{pid}.json')
            case = next(c for c in data['cases'] if c['output'] == output)
            samples.extend(s for s in case['samples'] if not s['warmup'])
        mean, se = quality(pid, scenario)
        cap = point.get('target_e2e_ms')
        rows.append({'label': point['label'], 'policy': pid, 'scenario': scenario,
            'family': point['family'], 'latency_fraction': point.get('latency_fraction'),
            'original_requested_linear_ms': point['requested_linear_limit_ms'],
            'minimum_linear_ms': point['minimum_linear_ms'],
            'effective_linear_limit_ms': point['linear_limit_ms'],
            'relaxed_infeasible': point['relaxed_infeasible'],
            'predicted_linear_ms': point['predicted_linear_ms'],
            'test_delta_nll': mean, 'paired_se': se,
            'median_e2e_ms': statistics.median(s['e2e_ms'] for s in samples),
            'median_cycle_ms': statistics.median(s['cycle_ms'] for s in samples),
            'target_e2e_ms': cap, 'generations': len(samples),
            'e2e_violations': sum(s['e2e_ms'] > cap for s in samples) if cap else None,
            'cycle_violations': sum(s['cycle_ms'] > cap for s in samples) if cap else None})
    with (RUN / 'results/hawq_latency_controls.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(RUN / 'results/hawq_controls_summary.json', {
        'labels': len(rows), 'unique_policy_scenarios': len({(r['policy'], r['scenario']) for r in rows}),
        'original_infeasible': sum(r['relaxed_infeasible'] for r in rows[:len(original)]),
        'additional_feasible': sum(not r['relaxed_infeasible'] for r in rows[len(original):]),
        'scope': 'No test-based choice. All three fixed latency budgets retained. NLL is measured quality, not a constraint in the HAWQ objective. Original relaxed limits remain explicit.'})


if __name__ == '__main__':
    main()
