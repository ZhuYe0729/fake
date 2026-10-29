"""Freeze fresh-profile allocations and validation rules before new quality labels."""
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from common import RUN, write_json
from daso.optimization import Action, ModuleChoices, solve_budget
from exact_solver import solve


def read(path):
    return json.loads(path.read_text())


def main():
    assert not (RUN / 'configs/reanalysis_pool.json').exists()
    methods = ['dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4', 'w4a16_ours', 'sq_int8', 'fp8']
    budgets = [.005, .01, .02, .04, .07]
    profile = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
               for r in csv.DictReader((RUN / 'results/graph_action_profile.csv').open())}
    template = read(RUN / 'configs/policies/uniform_dense_bf16.json')
    names = list(template['method_map'])
    catalog, signatures, points = {}, {}, []
    matrices, raw_exact = {}, {}

    def add(label, mapping, scenario, family, **metadata):
        signature = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        pid = signatures.setdefault(signature, label)
        if pid not in catalog:
            path = RUN / f'configs/policies/{pid}.json'
            if path.exists():
                assert read(path)['method_map'] == mapping
            else:
                write_json(path, {**template, 'policy_id': pid, 'scenario': scenario, 'method_map': mapping})
            catalog[pid] = {'signature': signature, 'scenarios': [],
                            'daso_eligible': all(v in methods for pair in mapping.values() for v in pair.values())}
        if scenario not in catalog[pid]['scenarios']:
            catalog[pid]['scenarios'].append(scenario)
        points.append({'label': label, 'policy_id': pid, 'scenario': scenario, 'family': family, **metadata})
        return pid

    # Retain every original main-table map, including unfavorable comparators.
    fixed = read(RUN.parent / '013/configs/design.json')
    for row in fixed['rows']:
        for scenario in ['prefill_only', 'prefill_decode']:
            pid = row[scenario]['policy']
            add(pid, read(RUN / f'configs/policies/{pid}.json')['method_map'], scenario,
                'fixed main comparator', main_method=row['method'])

    def mapping(indices, phases):
        return {n: {'prefill_method': methods[indices[i]],
                    'decode_method': methods[indices[len(names) + i]] if len(phases) == 2 else methods[indices[i]]}
                for i, n in enumerate(names)}

    for scenario in ['prefill_only', 'prefill_decode']:
        source = read(RUN.parent / f'005/data/actions_{scenario}.json')
        assert source['names'] == names and source['methods'] == methods
        phases = source['phases']
        q, bias = np.array(source['quality']), source['bias']
        t = np.array([[profile[n.rsplit('.', 1)[-1], method, 16384 if phase == 'prefill' else 8]
                       * (1 if phase == 'prefill' else 63) for method in methods]
                      for phase in phases for n in names])
        matrices[scenario] = {'quality': q.tolist(), 'latency_ms': t.tolist(), 'bias': bias,
                              'names': names, 'methods': methods, 'phases': phases}
        write_json(RUN / f'data/graph_actions_{scenario}.json', matrices[scenario])
        for budget in budgets:
            result = solve(q, t, budget - bias)
            raw_exact[scenario, budget] = result
            add(f'graph_exact7_{scenario}_q{budget:g}', mapping(result['indices'], phases), scenario,
                'fresh exact surrogate', quality_budget=budget, predicted_delta_nll=bias + result['quality'],
                predicted_linear_ms=result['latency_ms'], solver={k: v for k, v in result.items() if k != 'indices'})
        for fraction in [.95, .80, .65]:
            cap = float(t[:, 0].sum()) * fraction
            result = solve(q, t, cap, inverse=True)
            add(f'graph_inverse_{scenario}_r{fraction:g}', mapping(result['indices'], phases), scenario,
                'fresh inverse surrogate', linear_latency_cap_ms=cap,
                predicted_delta_nll=bias + result['quality'], predicted_linear_ms=result['latency_ms'])
        for count, dp_budgets in [(7, [.01, .02, .04, .07]),
                                  (5, [.02] if scenario == 'prefill_only' else [.07])]:
            modules = []
            for i, name in enumerate(names):
                actions = []
                for pre in range(count):
                    for dec in range(count):
                        if len(phases) == 1 and pre != dec:
                            continue
                        quality = q[i, pre] + (q[len(names) + i, dec] if len(phases) == 2 else 0.)
                        latency = t[i, pre] + (t[len(names) + i, dec] if len(phases) == 2 else 0.)
                        actions.append(Action(methods[pre], methods[dec], float(quality), float(latency)))
                modules.append(ModuleChoices(name, tuple(actions)))
            maximum = sum(max(a.quality_cost for a in m.actions) for m in modules)
            for budget in dp_budgets:
                result = solve_budget(modules, budget - bias, max_quality=maximum)
                maps = {n: {'prefill_method': a.prefill_method, 'decode_method': a.decode_method}
                        for n, a in zip(names, result.actions, strict=True)}
                add(f'graph_dp{count}_{scenario}_q{budget:g}', maps, scenario,
                    f'fresh original DP {count} actions', quality_budget=budget,
                    predicted_delta_nll=bias + result.quality_cost, predicted_linear_ms=result.latency_ms)
    for pre in ['dense_bf16', 'dense_nvfp4', 'sq_int8', 'fp8']:
        add('graph_phase_' + pre + '_w4a16',
            {n: {'prefill_method': pre, 'decode_method': 'w4a16_ours'} for n in names},
            'prefill_decode', 'phase-uniform anchor')

    # Original Hessian/weight-error objective, fresh measured costs; signed traces retained.
    base = RUN.parent / '001'
    hessian = {r['module']: r['mean_eigenvalue'] for r in read(base / 'results/hessian_trace_blockwise_p16_b4_s256.json')['records']}
    sensitivity = defaultdict(float)
    for filename in ['weight_quantization_errors.csv', 'marlin_int4_weight_errors.csv', 'fp4_weight_quantization_errors.csv']:
        for r in csv.DictReader((base / 'results' / filename).open()):
            n = r['module']; part = n.rsplit('.', 1)[-1]
            kind = 'qkv_proj' if part in ['q_proj', 'k_proj', 'v_proj'] else 'gate_up_proj' if part in ['gate_proj', 'up_proj'] else part
            sensitivity[n.rsplit('.', 1)[0] + '.' + kind, r['method']] += hessian[n] * float(r['weight_error_squared'])
    for scenario in ['prefill_only', 'prefill_decode']:
        for fmt in ['int4', 'fp4']:
            choices = ['marlin_int4' if fmt == 'int4' else 'dense_nvfp4', 'int8', 'dense_bf16']

            def dec(method):
                return 'w4a16_ours' if fmt == 'fp4' and method == 'dense_nvfp4' and scenario == 'prefill_decode' else method

            times = np.array([[profile[n.rsplit('.', 1)[-1], m, 16384] +
                (63 * profile[n.rsplit('.', 1)[-1], dec(m), 8] if scenario == 'prefill_decode' else 0.)
                for m in choices] for n in names])
            scores = np.array([[sensitivity[n, m] for m in choices] for n in names])
            minimum = float(times.min(1).sum())
            for budget in ([.01] if fmt == 'int4' else [.01, .07]):
                requested = raw_exact[scenario, budget]['latency_ms']
                cap = max(requested, minimum) + 1e-7
                matrix = lil_matrix((len(names) + 1, len(names) * 3))
                for i in range(len(names)):
                    matrix[i, i * 3:(i + 1) * 3] = 1
                matrix[-1] = times.reshape(-1)
                result = milp(scores.reshape(-1) / max(float(abs(scores).max()), 1e-12),
                    integrality=np.ones(scores.size), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix.tocsr(), np.r_[np.ones(len(names)), -np.inf],
                                                 np.r_[np.ones(len(names)), cap]),
                    options={'mip_rel_gap': 0.0})
                assert result.success, result.message
                selected = np.rint(result.x).astype(int).reshape(-1, 3)
                maps = {n: {'prefill_method': choices[j], 'decode_method': dec(choices[j])}
                        for n, j in zip(names, selected.argmax(1), strict=True)}
                add(f'graph_hawq_{fmt}_{scenario}_q{budget:g}', maps, scenario, 'fresh HAWQ PTQ adaptation',
                    paired_quality_budget=budget, requested_linear_limit_ms=requested, minimum_linear_ms=minimum,
                    relaxed_infeasible=requested < minimum - 1e-7, linear_limit_ms=cap,
                    predicted_linear_ms=float((times * selected).sum()), sensitivity_objective=float((scores * selected).sum()))
    pool = {'points': points, 'policies': catalog,
            'profile_sha256': hashlib.sha256((RUN / 'results/graph_action_profile.csv').read_bytes()).hexdigest()}
    write_json(RUN / 'configs/reanalysis_pool.json', pool)
    write_json(RUN / 'configs/selection_rules.json', {'quality_budgets': budgets,
        'inverse_latency_fractions': [.95, .80, .65], 'quality_guard': {'z': 2., 'additive_delta_nll': .005},
        'latency_guard_fraction': .05, 'speed_calibration_seed': 148015771,
        'speed_test_seed': 144983, 'quality_windows': 64, 'quality_batch': 8,
        'selection': 'Only independent validation NLL and calibration E2E; baseline BF16 guard is exact zero. Test labels do not choose candidates.',
        'quality_sources': 'Frozen WikiText validation/test windows from run005; same-map prior labels may be inherited with signature checks.',
        'pool_sha256': hashlib.sha256((RUN / 'configs/reanalysis_pool.json').read_bytes()).hexdigest(),
        'scope': 'New timing profile, unchanged quality surrogate; exact means additive surrogate only. Existing-map runtime probes are retained and not selection data.'})
    print('Frozen', len(points), 'labels and', len(catalog), 'unique maps', flush=True)


if __name__ == '__main__':
    main()
