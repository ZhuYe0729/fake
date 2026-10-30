"""HAWQ PTQ adaptation at the DASO proposals' matched additive latency caps."""
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from common import RUN, write_json


def main():
    output = RUN / 'results/hawq_allocation.json'
    assert not output.exists()
    trace_path = RUN / 'results/hessian_trace_p16_b4_s256.json'
    trace = json.loads(trace_path.read_text())
    assert trace['complete'] and len(trace['records']) == 64 * 7
    hessian = {row['module']: row['mean_eigenvalue'] for row in trace['records']}
    scores = defaultdict(float)
    with (RUN / 'results/weight_quantization_errors.csv').open() as stream:
        for row in csv.DictReader(stream):
            scores[row['fused_module'], row['method']] += hessian[row['module']] * float(row['weight_error_squared'])
    with (RUN / 'results/graph_action_profile.csv').open() as stream:
        latency = {(row['module_type'], row['method'], int(row['m'])): float(row['latency_ms'])
                   for row in csv.DictReader(stream)}
    template = json.loads((RUN / 'configs/policies/uniform_dense_bf16.json').read_text())
    names = list(template['method_map'])
    methods = ['dense_nvfp4', 'int8', 'dense_bf16']
    sensitivities = np.array([[scores[name, method] for method in methods] for name in names])
    assert np.isfinite(sensitivities).all()
    design = json.loads((RUN / 'configs/design.json').read_text())
    points = []
    for scenario in ['prefill_only', 'prefill_decode']:
        def decode(method):
            return 'w4a16_ours' if method == 'dense_nvfp4' and scenario == 'prefill_decode' else method
        times = np.array([[latency[name.rsplit('.', 1)[-1], method, 16384] +
            (63 * latency[name.rsplit('.', 1)[-1], decode(method), 8] if scenario == 'prefill_decode' else 0.)
            for method in methods] for name in names])
        proposals = [point for point in json.loads((RUN / 'configs/daso_selected.json').read_text())['points'] if point['scenario']==scenario]
        for proposal in proposals:
            if 'policy_id' not in proposal:
                continue
            requested = proposal['predicted_linear_ms']
            minimum = float(times.min(1).sum())
            cap = max(requested, minimum) + 1e-7
            matrix = lil_matrix((len(names) + 1, len(names) * len(methods)))
            for index in range(len(names)):
                matrix[index, index * len(methods):(index + 1) * len(methods)] = 1
            matrix[-1] = times.reshape(-1)
            result = milp(sensitivities.reshape(-1) / max(float(abs(sensitivities).max()), 1e-12),
                integrality=np.ones(sensitivities.size), bounds=Bounds(0, 1),
                constraints=LinearConstraint(matrix.tocsr(), np.r_[np.ones(len(names)), -np.inf],
                    np.r_[np.ones(len(names)), cap]), options={'mip_rel_gap': 0.0})
            assert result.success, result.message
            choice = np.rint(result.x).astype(int).reshape(len(names), len(methods))
            assert (choice.sum(1) == 1).all() and float((times * choice).sum()) <= cap + 1e-7
            policy_id = f"hawq32_{scenario}_matched_q{proposal['quality_budget']:g}"
            mapping = {name: {'prefill_method': methods[index], 'decode_method': decode(methods[index])}
                for name, index in zip(names, choice.argmax(1), strict=True)}
            path = RUN / f'configs/policies/{policy_id}.json'
            write_json(path, {**template, 'policy_id': policy_id, 'scenario': scenario, 'method_map': mapping})
            design['policies'][policy_id] = {'display': policy_id, 'scenario': scenario,
                'map_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            points.append({'policy_id': policy_id, 'scenario': scenario, 'paired_daso': proposal['policy_id'],
                'paired_daso_budget': proposal['quality_budget'], 'requested_linear_limit_ms': requested,
                'minimum_linear_ms': minimum, 'linear_limit_ms': cap, 'relaxed_infeasible': requested < minimum - 1e-7,
                'predicted_linear_ms': float((times * choice).sum()),
                'sensitivity_objective': float((sensitivities * choice).sum())})
    write_json(output, {'points': points, 'trace_source_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'negative_trace_blocks': sum(row['trace_mean'] < 0 for row in trace['records']),
        'trace_transform': 'signed, unchanged', 'prefill_choices': methods,
        'decode_fp4_representation': 'w4a16_ours',
        'scope': 'Qwen3 NLL Hessian/weight-error PTQ adaptation; not the original CNN QAT/TVM pipeline. '
                 'Matches additive latency caps, not true NLL budgets or measured E2E. DASO has eight candidates.'})
    write_json(RUN / 'configs/design.json', design)


if __name__ == '__main__':
    main()
