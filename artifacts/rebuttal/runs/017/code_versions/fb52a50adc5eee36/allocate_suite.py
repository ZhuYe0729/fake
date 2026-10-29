"""Identical candidate/cost/cap ILPs; only the quality objective changes."""
import csv
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

from common import RUN, SPEC, write_json
from exact_solver import solve as forward_solve


def read(path):
    return json.loads(path.read_text())


def solve(scores, costs, limit):
    n, k = scores.shape
    minimum = float(costs.min(1).sum())
    if limit < minimum - 1e-7:
        return {'feasible': False, 'minimum_linear_ms': minimum}
    matrix = lil_matrix((n+1, n*k))
    for i in range(n):
        matrix[i, i*k:(i+1)*k] = 1
    matrix[-1] = costs.reshape(-1)
    started = time.perf_counter()
    score_scale = max(float(abs(scores).max()), 1e-12)
    normalized = scores.reshape(-1)/score_scale
    result = milp(normalized,
        integrality=np.ones(scores.size), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), np.r_[np.ones(n), -np.inf], np.r_[np.ones(n), limit+1e-7]),
        options={'mip_rel_gap': 0., 'time_limit': 120.})
    assert result.success, result.message
    optimum = float(result.fun)
    # Equal rule for both proxies: choose the least costly quality-optimal point.
    second_matrix = vstack([matrix.tocsr(), normalized.reshape(1, -1)], format='csr')
    result = milp(costs.reshape(-1), integrality=np.ones(scores.size), bounds=Bounds(0, 1),
        constraints=LinearConstraint(second_matrix, np.r_[np.ones(n), -np.inf, -np.inf],
            np.r_[np.ones(n), limit+1e-7, optimum+1e-8]),
        options={'mip_rel_gap': 0., 'time_limit': 120.})
    assert result.success, result.message
    onehot = np.rint(result.x).astype(int).reshape(n, k)
    assert np.all(onehot.sum(1) == 1) and float((onehot*costs).sum()) <= limit+1e-6
    return {'feasible': True, 'indices': onehot.argmax(1).tolist(),
        'predicted_linear_ms': float((onehot*costs).sum()), 'objective': float((onehot*scores).sum()),
        'seconds': time.perf_counter()-started, 'optimal': result.status == 0,
        'quality_optimum': optimum*score_scale, 'tie_break': 'minimum predicted latency within normalized quality optimum +1e-8',
        'mip_gap': float(result.mip_gap)}


def main():
    assert not (RUN / 'configs/allocation_pool.json').exists()
    frozen = read(RUN / 'configs/frozen_design.json')
    names, methods = frozen['names'], frozen['methods']
    profile = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
               for r in csv.DictReader((RUN / 'reference/graph_action_profile.csv').open())}
    traces = {r['module']: r['mean_eigenvalue'] for r in read(
        RUN / 'results/hessian_trace.json')['records']}
    sensitivity = {mode: defaultdict(float) for mode in ['signed', 'clip_zero', 'absolute']}
    for row in csv.DictReader((RUN / 'data/weight_errors_runtime.csv').open()):
        h = traces[row['module']]
        for mode, value in [('signed', h), ('clip_zero', max(h, 0.)), ('absolute', abs(h))]:
            sensitivity[mode][row['fused_module'], row['method']] += value * float(row['weight_error_squared'])
    catalog, signatures, points = {}, {}, []
    template = read(RUN / 'configs/policies/uniform_dense_bf16.json')

    def add(label, mapping, scene, family, selectable=False, **metadata):
        digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        pid = signatures.setdefault(digest, label)
        if pid not in catalog:
            write_json(RUN / f'configs/policies/{pid}.json', {**template, 'policy_id': pid, 'scenario': scene, 'method_map': mapping})
            catalog[pid] = {'signature': digest, 'scenarios': [], 'selection_scenarios': []}
        if scene not in catalog[pid]['scenarios']:
            catalog[pid]['scenarios'].append(scene)
        if selectable and scene not in catalog[pid]['selection_scenarios']:
            catalog[pid]['selection_scenarios'].append(scene)
        points.append({'label': label, 'policy_id': pid, 'scenario': scene, 'family': family, **metadata})

    for method in ['dense_bf16', 'int8', 'dense_nvfp4', 'w4a16_ours', 'sq_int8', 'fp8', 'sparse_bf16', 'sparse_nvfp4']:
        for scene in ['prefill_only', 'prefill_decode']:
            add('uniform_'+method, {n: {'prefill_method': method, 'decode_method': method} for n in names},
                scene, 'uniform anchor', selectable=True)

    for pre, dec in read(RUN / 'configs/proposal_rules.json')['fixed_uniform_phase_pairs']:
        add('phase_'+pre+'_'+dec, {n: {'prefill_method':pre, 'decode_method':dec} for n in names},
            'prefill_decode', 'uniform phase anchor', selectable=True, phase_mode='fixed global pair')

    for original in read(RUN / 'configs/original_main_policies.json'):
        add(original['label'], read(__import__('pathlib').Path(original['policy_json']))['method_map'],
            original['scenario'], 'original DASO-5', selectable=True, original_source=original['source'],
            original_sha256=original['source_sha256'])

    calibration = read(RUN / 'reference/calibration_uniform_dense_bf16.json')
    for scene in ['prefill_only', 'prefill_decode']:
        actions = read(RUN / f'data/actions_{scene}.json')
        q = np.array(actions['quality'])
        phases = actions['phases']
        base = next(c['medians']['e2e_ms'] for c in calibration['cases'] if c['output'] == (1 if scene == 'prefill_only' else 64))
        baseline_linear = sum(profile[n.rsplit('.', 1)[-1], 'dense_bf16', 16384] +
            (63*profile[n.rsplit('.', 1)[-1], 'dense_bf16', 8] if len(phases) == 2 else 0.) for n in names)
        offset = base-baseline_linear
        modes = ['same_operator'] if scene == 'prefill_only' else ['same_operator', 'phase_pair']
        for mode in modes:
            pairs = [('dense_nvfp4', 'w4a16_ours' if mode == 'phase_pair' else 'dense_nvfp4'),
                     ('int8', 'int8'), ('dense_bf16', 'dense_bf16')]
            costs = np.array([[profile[n.rsplit('.', 1)[-1], pre, 16384] +
                (63*profile[n.rsplit('.', 1)[-1], dec, 8] if len(phases) == 2 else 0.) for pre, dec in pairs] for n in names])
            score = np.array([[q[i, methods.index(pre)] +
                (q[len(names)+i, methods.index(dec)] if len(phases) == 2 else 0.) for pre, dec in pairs] for i in range(len(names))])
            hs = {variant: np.array([[sensitivity[variant][n, pre] for pre, _ in pairs] for n in names])
                  for variant in sensitivity}
            write_json(RUN / f'data/matched_matrix_{scene}_{mode}.json', {'pairs': pairs, 'names': names,
                'costs': costs.tolist(), 'daso_quality': score.tolist(),
                'hessian': {k: v.tolist() for k, v in hs.items()}, 'bf16_e2e_offset_ms': offset})

            def mapping(indices):
                return {n: {'prefill_method': pairs[j][0], 'decode_method': pairs[j][1]}
                        for n, j in zip(names, indices, strict=True)}

            for fraction in frozen['latency_fractions']:
                cap = base*fraction/1.05-offset
                goals = [('daso', score, 'daso matched'), ('hawq_signed', hs['signed'], 'hawq matched')]
                if fraction == .80:
                    goals += [('hawq_'+v, hs[v], 'hawq robustness') for v in ['clip_zero', 'absolute']]
                for allocator, objective, family in goals:
                    result = solve(objective, costs, cap)
                    label = f'{allocator}_{scene}_{mode}_r{fraction:g}'
                    metadata = {'phase_mode': mode, 'allocator': allocator, 'latency_fraction': fraction,
                        'actual_e2e_target_ms': base*fraction, 'linear_limit_ms': cap,
                        'minimum_linear_ms': float(costs.min(1).sum()), 'solver': result}
                    if not result['feasible']:
                        points.append({'label': label, 'policy_id': None, 'scenario': scene, 'family': family, **metadata})
                        continue
                    add(label, mapping(result['indices']), scene, family, selectable=allocator == 'daso',
                        predicted_delta_nll=actions['bias']+float(score[np.arange(len(names)), result['indices']].sum()), **metadata)
            minimum = float(costs.min(1).sum())
            a, b = solve(score, costs, minimum), solve(hs['signed'], costs, minimum)
            assert a['feasible'] and b['feasible']
            assert abs(a['predicted_linear_ms']-b['predicted_linear_ms']) < 1e-6
            add(f'hardware_minimum_{scene}_{mode}', mapping(costs.argmin(1)), scene, 'hardware anchor',
                selectable=True, phase_mode=mode, minimum_linear_ms=minimum,
                endpoint_daso_indices=a['indices'], endpoint_hawq_indices=b['indices'])

        if scene == 'prefill_decode':
            pairs = [(pre, dec) for pre in ['dense_bf16', 'int8', 'dense_nvfp4']
                     for dec in ['dense_bf16', 'int8', 'dense_nvfp4', 'w4a16_ours']]
            costs = np.array([[profile[n.rsplit('.', 1)[-1], pre, 16384] + 63*profile[n.rsplit('.', 1)[-1], dec, 8]
                               for pre, dec in pairs] for n in names])
            score = np.array([[q[i, methods.index(pre)] + q[len(names)+i, methods.index(dec)]
                               for pre, dec in pairs] for i in range(len(names))])
            for fraction in frozen['latency_fractions']:
                cap = base*fraction/1.05-offset
                result = solve(score, costs, cap)
                if not result['feasible']:
                    points.append({'label': f'daso_independent_{scene}_r{fraction:g}', 'policy_id': None,
                        'scenario': scene, 'family': 'daso independent matched formats',
                        'latency_fraction': fraction, 'actual_e2e_target_ms': base*fraction,
                        'linear_limit_ms': cap, 'solver': result})
                    continue
                mapping = {n: {'prefill_method': pairs[j][0], 'decode_method': pairs[j][1]}
                           for n, j in zip(names, result['indices'], strict=True)}
                add(f'daso_independent_{scene}_r{fraction:g}', mapping, scene, 'daso independent matched formats',
                    selectable=True, latency_fraction=fraction, actual_e2e_target_ms=base*fraction,
                    linear_limit_ms=cap, solver=result)
        full_cost = np.array([[profile[n.rsplit('.', 1)[-1], method, 16384 if phase == 'prefill' else 8] *
            (1 if phase == 'prefill' else 63) for method in methods] for phase in phases for n in names])
        for budget in frozen['quality_budgets']:
            if budget-actions['bias'] < float(q.min(1).sum())-1e-9:
                points.append({'label': f'daso8_exact_{scene}_q{budget:g}', 'policy_id': None,
                    'scenario': scene, 'family': 'daso eight forward', 'quality_budget': budget,
                    'reason': 'Quality surrogate infeasible; BF16 remains a known identity anchor.'})
                continue
            result = forward_solve(q, full_cost, budget-actions['bias'])
            mapping = {n: {'prefill_method': methods[result['indices'][i]],
                'decode_method': methods[result['indices'][i+len(names)]] if len(phases) == 2 else methods[result['indices'][i]]}
                for i, n in enumerate(names)}
            add(f'daso8_exact_{scene}_q{budget:g}', mapping, scene, 'daso eight forward', selectable=True,
                quality_budget=budget, predicted_delta_nll=actions['bias']+result['quality'], solver=result)
    pool = {'points': points, 'policies': catalog, 'design_sha256': hashlib.sha256((RUN / 'configs/frozen_design.json').read_bytes()).hexdigest(),
        'scope': 'All allocation points frozen before validation/confirmation. Matched costs and constraints are identical; only objective differs. Phase modes prespecified. No silent infeasibility relaxation.'}
    write_json(RUN / 'configs/allocation_pool.json', pool)
    jobs = [{'job_id': pid, 'policy': pid, 'scenarios': p['scenarios']} for pid, p in catalog.items()]
    write_json(RUN / 'configs/validation_jobs.json', jobs)
    write_json(RUN / 'configs/confirmation_jobs.json', jobs)
    print('Frozen', len(points), 'points,', len(catalog), 'unique policies.', flush=True)


if __name__ == '__main__':
    main()
