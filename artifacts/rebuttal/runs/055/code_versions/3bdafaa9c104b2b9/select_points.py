"""Validation-only closure of the fixed two-budget DASO representatives."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import time

from common import RUN, write_json
from policy_runtime import quality


def prepare():
    target = RUN / 'configs/validation_pool.json'
    assert not target.exists()
    points = [{'policy_id': policy, 'scenario': scenario} for policy in
        ['uniform_dense_bf16', 'uniform_int8', 'uniform_sq_int8', 'uniform_fp8']
        for scenario in ['prefill_only', 'prefill_decode']]
    for scenario in ['prefill_only', 'prefill_decode']:
        proposals = json.loads((RUN / f'results/quality_model/{scenario}/proposals.json').read_text())['points']
        points += [{'policy_id': point['policy_id'], 'scenario': scenario}
                   for point in proposals if 'policy_id' in point]
    rules_path = RUN / 'configs/selection_rules.json'
    rules = json.loads(rules_path.read_text())
    write_json(target, {'points': points, 'budgets': rules['representative_quality_budgets'],
        'z': rules['validation_guard']['paired_standard_errors'],
        'margin': rules['validation_guard']['additive_margin_nats_per_token'],
        'selection_rules_sha256': hashlib.sha256(rules_path.read_bytes()).hexdigest(),
        'rule': 'Choose minimum additive profile cost among validation-feasible policies; BF16 is always eligible. '
                'Uniforms are selection anchors, not additional formal baseline rows. No test/downstream score is read.'})


def evaluate():
    pool = json.loads((RUN / 'configs/validation_pool.json').read_text())
    done = set()
    for point in pool['points']:
        policy = point['policy_id']
        if policy in done:
            continue
        quality(policy, 'validation', f'validation_{policy}_{time.time_ns()}')
        done.add(policy)


def select():
    target = RUN / 'configs/daso_selected.json'
    assert not target.exists()
    pool_path = RUN / 'configs/validation_pool.json'
    pool = json.loads(pool_path.read_text())
    design = json.loads((RUN / 'configs/design.json').read_text())
    with (RUN / 'results/graph_action_profile.csv').open() as stream:
        latency = {(row['module_type'], row['method'], int(row['m'])): float(row['latency_ms'])
                   for row in csv.DictReader(stream)}
    candidates, selected = [], []
    for scenario in ['prefill_only', 'prefill_decode']:
        case = next(case for case in design['workloads'] if case['scenario'] == scenario)
        baseline = json.loads((RUN / f"results/quality/validation/uniform_dense_bf16/{case['label']}.json").read_text())
        rows = []
        for point in [point for point in pool['points'] if point['scenario'] == scenario]:
            policy = point['policy_id']
            path = RUN / f"results/quality/validation/{policy}/{case['label']}.json"
            result = json.loads(path.read_text())
            assert result['complete'] and len(result['blocks']) == len(baseline['blocks']) == 100
            assert result['sample_sha256'] == baseline['sample_sha256']
            differences = [a['avg_nll'] - b['avg_nll'] for a, b in zip(result['blocks'], baseline['blocks'], strict=True)]
            mean = statistics.mean(differences)
            se = statistics.stdev(differences) / math.sqrt(len(differences))
            mapping = json.loads((RUN / f'configs/policies/{policy}.json').read_text())['method_map']
            cost = sum(latency[name.rsplit('.', 1)[-1], pair['prefill_method'], 16384] +
                (63 * latency[name.rsplit('.', 1)[-1], pair['decode_method'], 8] if scenario == 'prefill_decode' else 0.)
                for name, pair in mapping.items())
            rows.append({**point, 'delta_nll': mean, 'paired_se': se, 'predicted_linear_ms': cost,
                'guarded_delta_nll': 0. if policy == 'uniform_dense_bf16' else mean + pool['z'] * se + pool['margin'],
                'validation_result_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        candidates += rows
        for budget in pool['budgets']:
            feasible = [row for row in rows if row['guarded_delta_nll'] <= budget]
            winner = min(feasible, key=lambda row: (row['predicted_linear_ms'], row['policy_id']))
            selected.append({**winner, 'quality_budget': budget, 'feasible_candidates': len(feasible)})
    write_json(target, {'points': selected, 'all_validation_candidates': candidates,
        'pool_sha256': hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        'scope': 'Validation closure with paired mean+2SE+0.005 margin; fixed public text is shared. '
                 'No test/downstream selection and no hard quality/E2E guarantee.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'evaluate', 'select'])
    args = parser.parse_args()
    {'prepare': prepare, 'evaluate': evaluate, 'select': select}[args.action]()
