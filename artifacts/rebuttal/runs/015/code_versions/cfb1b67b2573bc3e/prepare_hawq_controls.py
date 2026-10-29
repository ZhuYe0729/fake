"""Freeze feasible HAWQ latency controls using hardware/calibration data only."""
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from common import RUN, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    destination = RUN / 'configs/hawq_latency_controls.json'
    if destination.exists():
        return
    template = read(RUN / 'configs/policies/uniform_dense_bf16.json')
    names = list(template['method_map'])
    profile = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
               for r in csv.DictReader((RUN / 'results/graph_action_profile.csv').open())}
    base = RUN.parent / '001/results'
    hessian = {r['module']: r['mean_eigenvalue'] for r in
               read(base / 'hessian_trace_blockwise_p16_b4_s256.json')['records']}
    sensitivity = defaultdict(float)
    for filename in ['weight_quantization_errors.csv', 'marlin_int4_weight_errors.csv',
                     'fp4_weight_quantization_errors.csv']:
        for r in csv.DictReader((base / filename).open()):
            n = r['module']
            part = n.rsplit('.', 1)[-1]
            kind = ('qkv_proj' if part in ['q_proj', 'k_proj', 'v_proj'] else
                    'gate_up_proj' if part in ['gate_proj', 'up_proj'] else part)
            sensitivity[n.rsplit('.', 1)[0] + '.' + kind, r['method']] += (
                hessian[n] * float(r['weight_error_squared']))
    signatures = {}
    for path in sorted((RUN / 'configs/policies').glob('*.json')):
        mapping = read(path)['method_map']
        signatures.setdefault(json.dumps(mapping, sort_keys=True), path.stem)
    calibration_path = RUN / 'results/measurements/calibration_uniform_dense_bf16.json'
    calibration = read(calibration_path)
    points = []
    for scenario in ['prefill_only', 'prefill_decode']:
        output = 1 if scenario == 'prefill_only' else 64
        actual = next(c['medians']['e2e_ms'] for c in calibration['cases'] if c['output'] == output)
        bf16_linear = sum(profile[n.rsplit('.', 1)[-1], 'dense_bf16', 16384] +
            (63 * profile[n.rsplit('.', 1)[-1], 'dense_bf16', 8] if output == 64 else 0.) for n in names)
        offset = actual - bf16_linear
        for fmt in ['int4', 'fp4']:
            choices = ['marlin_int4' if fmt == 'int4' else 'dense_nvfp4', 'int8', 'dense_bf16']

            def decode(method):
                return 'w4a16_ours' if output == 64 and method == 'dense_nvfp4' else method

            times = np.array([[profile[n.rsplit('.', 1)[-1], m, 16384] +
                (63 * profile[n.rsplit('.', 1)[-1], decode(m), 8] if output == 64 else 0.)
                for m in choices] for n in names])
            scores = np.array([[sensitivity[n, m] for m in choices] for n in names])
            matrix = lil_matrix((len(names) + 1, len(names) * 3))
            for i in range(len(names)):
                matrix[i, i * 3:(i + 1) * 3] = 1
            matrix[-1] = times.reshape(-1)
            for fraction in [.95, .80, .65]:
                requested = actual * fraction / 1.05 - offset
                minimum = float(times.min(1).sum())
                cap = max(requested, minimum) + 1e-7
                result = milp(scores.reshape(-1) / max(float(abs(scores).max()), 1e-12),
                    integrality=np.ones(scores.size), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix.tocsr(), np.r_[np.ones(len(names)), -np.inf],
                        np.r_[np.ones(len(names)), cap]), options={'mip_rel_gap': 0.0})
                assert result.success, result.message
                selected = np.rint(result.x).astype(int).reshape(-1, 3)
                mapping = {n: {'prefill_method': choices[j], 'decode_method': decode(choices[j])}
                    for n, j in zip(names, selected.argmax(1), strict=True)}
                label = f'graph_hawq_{fmt}_{scenario}_r{fraction:g}'
                pid = signatures.setdefault(json.dumps(mapping, sort_keys=True), label)
                path = RUN / f'configs/policies/{pid}.json'
                if not path.exists():
                    write_json(path, {**template, 'policy_id': pid, 'scenario': scenario, 'method_map': mapping})
                points.append({'label': label, 'policy_id': pid, 'scenario': scenario,
                    'family': 'HAWQ feasible latency control', 'format': fmt, 'latency_fraction': fraction,
                    'baseline_calibration_e2e_ms': actual, 'target_e2e_ms': actual * fraction,
                    'bf16_additive_offset_ms': offset, 'requested_linear_limit_ms': requested,
                    'linear_limit_ms': cap, 'minimum_linear_ms': minimum,
                    'relaxed_infeasible': requested < minimum - 1e-7,
                    'predicted_linear_ms': float((times * selected).sum()),
                    'sensitivity_objective': float((scores * selected).sum())})
    write_json(destination, {'points': points, 'main_fraction': .80,
        'calibration_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        'selection_sha256': hashlib.sha256((RUN / 'configs/selected_reanalysis.json').read_bytes()).hexdigest(),
        'scope': 'Baseline-only amendment after DASO selection. No new test labels opened. Signed Hessian convention unchanged. All three preregistered inverse fractions reported; .80 is the fixed main row, not selected by test. Offset and 5% margin are an additive timing proxy, not a true E2E guarantee.'})
    print('HAWQ controls', len(points), 'relaxed', sum(p['relaxed_infeasible'] for p in points), flush=True)


if __name__ == '__main__':
    main()
