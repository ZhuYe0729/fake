"""Original HAWQ-v3 sensitivity objective and latency constraint, solved by HiGHS ILP."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from common import RUN, write_json


def solve(sensitivity: np.ndarray, latency: np.ndarray, limit: float) -> np.ndarray:
    modules, methods = sensitivity.shape
    matrix = lil_matrix((modules + 1, modules * methods))
    for index in range(modules):
        matrix[index, index * methods:(index + 1) * methods] = 1
    matrix[-1] = latency.reshape(-1)
    constraint = LinearConstraint(matrix.tocsr(), np.r_[np.ones(modules), -np.inf],
                                 np.r_[np.ones(modules), limit])
    scale = max(float(np.abs(sensitivity).max()), 1e-12)
    result = milp(sensitivity.reshape(-1) / scale,
                  integrality=np.ones(modules * methods), bounds=Bounds(0, 1),
                  constraints=constraint, options={'mip_rel_gap': 0.0})
    if not result.success:
        raise RuntimeError(result.message)
    selected = np.rint(result.x).astype(int).reshape(modules, methods)
    if not np.all(selected.sum(1) == 1) or (selected * latency).sum() > limit + 1e-7:
        raise RuntimeError('ILP solution violates the original constraints')
    return selected.argmax(1)


def fused(name: str) -> str:
    for suffix in ['q_proj', 'k_proj', 'v_proj']:
        if name.endswith('.' + suffix):
            return name.rsplit('.', 1)[0] + '.qkv_proj'
    for suffix in ['gate_proj', 'up_proj']:
        if name.endswith('.' + suffix):
            return name.rsplit('.', 1)[0] + '.gate_up_proj'
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--low-method', choices=['int4','dense_nvfp4'], default='int4')
    parser.add_argument('--absolute-trace', action='store_true',
                        help='Explicit robustness variant; never silently replace the signed original objective.')
    args = parser.parse_args()
    started = time.perf_counter()
    traces = json.loads(args.trace.read_text())
    if not traces['complete'] or len(traces['records']) != 36 * 7:
        raise ValueError('incomplete Hessian trace calibration')
    coefficients = {row['module']: row['mean_eigenvalue'] for row in traces['records']}
    with (RUN / 'results/weight_quantization_errors.csv').open() as handle:
        errors = list(csv.DictReader(handle))
    if args.low_method == 'dense_nvfp4':
        with (RUN / 'results/fp4_weight_quantization_errors.csv').open() as handle:
            errors.extend(csv.DictReader(handle))
    costs = defaultdict(float)
    for row in errors:
        coefficient = coefficients[row['module']]
        if args.absolute_trace:
            coefficient = abs(coefficient)
        costs[fused(row['module']), row['method']] += coefficient * float(row['weight_error_squared'])
    with (RUN / 'results/action_profile.csv').open() as handle:
        profile = {(row['module_type'], row['method'], int(row['m'])): row for row in csv.DictReader(handle)}
    names = sorted({name for name, _ in costs})
    methods = [args.low_method, 'int8']
    sensitivity = np.array([[costs[name, method] for method in methods] for name in names])
    outputs = []
    for scenario in ['prefill_only', 'prefill_decode']:
        latency = []
        for name in names:
            kind = '.'.join(name.split('.')[-2:])
            values = []
            for method in methods:
                pre, dec = profile[kind, method, 16384], profile[kind, method, 8]
                if pre['supported'] != 'True' or dec['supported'] != 'True':
                    raise ValueError(f'unsupported HAWQ action {kind}/{method}')
                values.append(float(pre['latency_ms']) +
                    (63 * float(dec['latency_ms']) if scenario == 'prefill_decode' else 0))
            latency.append(values)
        latency = np.array(latency)
        minimum, maximum = latency.min(1).sum(), latency.max(1).sum()
        signatures = set()
        for index, fraction in enumerate([0, .1, .25, .5, .75, 1]):
            limit = float(minimum + fraction * (maximum - minimum) + 1e-7)
            choice = solve(sensitivity, latency, limit)
            signature = tuple(choice.tolist())
            if signature in signatures:
                continue
            signatures.add(signature)
            variant = ('hawq_fp4' if args.low_method == 'dense_nvfp4' else 'hawq') + ('_abs' if args.absolute_trace else '')
            policy_id = f'{variant}_{scenario}_{index:02d}'
            policy = {'policy_id': policy_id, 'scenario': scenario,
                'default_prefill_method': 'int8', 'default_decode_method': 'int8',
                'modules_to_not_convert': ['lm_head'],
                'method_map': {name: {'prefill_method': methods[selection],
                                     'decode_method': methods[selection]}
                               for name, selection in zip(names, choice, strict=True)}}
            write_json(RUN / f'configs/policies/{policy_id}.json', policy)
            outputs.append({'policy_id': policy_id, 'scenario': scenario,
                'latency_limit_ms': limit,
                'predicted_linear_ms': float(latency[np.arange(len(names)), choice].sum()),
                'hessian_sensitivity': float(sensitivity[np.arange(len(names)), choice].sum()),
                'low_method':args.low_method,'low_precision_modules': int((choice == 0).sum()), 'int8_modules': int((choice == 1).sum())})
    variant = ('hawq_fp4' if args.low_method == 'dense_nvfp4' else 'hawq') + ('_abs' if args.absolute_trace else '')
    write_json(RUN / f'results/{variant}_allocation.json', {
        'policies': outputs, 'trace_source': str(args.trace),
        'objective': 'min sum_i mean_trace(H_i) * ||Q_b(W_i)-W_i||_F^2',
        'trace_absolute_value': args.absolute_trace,
        'negative_trace_blocks': sum(row['trace_mean'] < 0 for row in traces['records']),
        'constraint': 'sum_i measured_Linear_latency_i <= latency_limit',
        'solver': 'scipy.optimize.milp / HiGHS, binary one-hot ILP',
        'adaptations': ['Qwen3 teacher-forced NLL instead of ImageNet classification',
            'fused QKV and gate/up choices tied for runtime compatibility',
            f'PTQ {args.low_method}/INT8 through common phase-vLLM, no original CNN QAT/TVM'],
        'solve_seconds': time.perf_counter() - started})
    print(outputs, flush=True)


if __name__ == '__main__':
    main()
