"""Fit fresh 32B quality models and solve their discrete additive proxies."""
import csv
import hashlib
import json
import time

import numpy as np
import torch

from common import RUN, write_json
from exact_solver import solve
from daso.modeling.quality import FUSED_TYPES, QualityCalibrator
import daso.modeling.quality as quality_module


def main():
    assert json.loads((RUN / 'results/prompt_label_reuse_check.json').read_text())['complete']
    manifest_path = RUN / 'configs/calibration_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    methods = manifest['methods']
    quality_module.METHODS = tuple(methods)
    entries = manifest['entries']
    design = json.loads((RUN / 'configs/design.json').read_text())
    baseline = json.loads((RUN / 'configs/policies/uniform_dense_bf16.json').read_text())
    names = list(baseline['method_map'])
    with (RUN / 'data/local_errors.csv').open() as stream:
        errors = {(row['module'], row['phase'], row['method']): float(row['output_rel_mse'])
                  for row in csv.DictReader(stream)}
    with (RUN / 'results/graph_action_profile.csv').open() as stream:
        latency = {(row['module_type'], row['method'], int(row['m'])): float(row['latency_ms'])
                   for row in csv.DictReader(stream)}
    for entry in entries:
        assert json.loads((RUN / f"results/calibration_complete/{entry['policy_id']}.json").read_text())['complete']
    for scenario in ['prefill_only', 'prefill_decode']:
        started = time.perf_counter()
        output = RUN / f'results/quality_model/{scenario}'
        assert not output.exists()
        output.mkdir(parents=True)
        case = next(case for case in design['workloads'] if case['scenario'] == scenario)
        phases = ['prefill'] if scenario == 'prefill_only' else ['prefill', 'decode']
        features, losses = [], []
        for entry in entries:
            policy_id = entry['policy_id']
            policy = json.loads((RUN / f'configs/policies/{policy_id}.json').read_text())
            feature = torch.zeros((2, len(methods), 4, len(FUSED_TYPES)), dtype=torch.float64)
            for name, pair in policy['method_map'].items():
                bucket, kind = int(name.split('.')[2]) * 4 // 64, FUSED_TYPES.index(name.rsplit('.', 1)[-1])
                for phase_index, phase in enumerate(phases):
                    method = pair[phase + '_method']
                    if method != 'dense_bf16':
                        feature[phase_index, methods.index(method), bucket, kind] += errors[name, phase, method]
            result = json.loads((RUN / f"results/quality/calibration/{policy_id}/{case['label']}.json").read_text())
            assert result['complete'] and result['sample_sha256'] == manifest['sample_sha256']
            features.append(feature)
            losses.append(sum(row['avg_nll'] for row in result['blocks']) / len(result['blocks']))
        x = torch.stack(features)
        y = torch.tensor(np.array(losses) - losses[0], dtype=torch.float64)
        train = torch.tensor([entry['split'] == 'train' for entry in entries])
        fitted = QualityCalibrator().fit(x, y, train)
        fitted.save(output / 'model.json')
        predicted = fitted.predict(x)
        write_json(output / 'predictions.json', [{**entry, 'actual_delta_nll': float(actual),
            'predicted_delta_nll': float(prediction)} for entry, actual, prediction in zip(entries, y, predicted, strict=True)])
        metrics = {split: {'policies': int(mask.sum()), 'mae': float((predicted[mask] - y[mask]).abs().mean()),
            'rmse': float((predicted[mask] - y[mask]).square().mean().sqrt())}
            for split, mask in [('train', train), ('holdout', ~train)]}
        write_json(output / 'metrics.json', {'splits': metrics, 'fit_seconds': time.perf_counter() - started,
            'calibration_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'scope': 'New32B local features and real phase/KV labels. Held-out policies share fixed train text; no test labels used.'})
        q, t = [], []
        for phase_index, phase in enumerate(phases):
            for name in names:
                bucket, kind = int(name.split('.')[2]) * 4 // 64, name.rsplit('.', 1)[-1]
                q.append([0. if method == 'dense_bf16' else errors[name, phase, method] *
                    float(fitted.coefficients[phase_index, methods.index(method), bucket, FUSED_TYPES.index(kind)]) /
                    fitted.feature_scale for method in methods])
                t.append([latency[kind, method, 16384 if phase == 'prefill' else 8] *
                          (1 if phase == 'prefill' else 63) for method in methods])
        actions = {'names': names, 'methods': methods, 'phases': phases, 'quality': q,
                   'latency_ms': t, 'bias': fitted.bias}
        write_json(RUN / f'data/actions_{scenario}.json', actions)
        points = []
        for budget in manifest['representative_quality_budgets']:
            if budget < fitted.bias:
                points.append({'budget': budget, 'status': 'proxy_bias_exceeds_budget'})
                continue
            first = solve(np.array(q), np.array(t), budget - fitted.bias)
            result = solve(np.array(q), np.array(t), first['latency_ms'] + 1e-7, inverse=True)
            assert result['quality'] <= budget - fitted.bias + 1e-8
            indices = result['indices']
            policy_id = f'daso32_{scenario}_q{budget:g}'
            mapping = {name: {'prefill_method': methods[indices[index]],
                'decode_method': methods[indices[len(names) + index]] if len(phases) == 2 else methods[indices[index]]}
                for index, name in enumerate(names)}
            path = RUN / f'configs/policies/{policy_id}.json'
            write_json(path, {**baseline, 'policy_id': policy_id, 'scenario': scenario, 'method_map': mapping})
            design['policies'][policy_id] = {'display': policy_id, 'scenario': scenario,
                'map_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            points.append({'policy_id': policy_id, 'quality_budget': budget,
                'predicted_delta_nll': fitted.bias + result['quality'], 'predicted_linear_ms': result['latency_ms'],
                'solver': {'optimal': first['optimal'] and result['optimal'],
                    'latency': {key: value for key, value in first.items() if key != 'indices'},
                    'quality_tie': {key: value for key, value in result.items() if key != 'indices'}}})
        write_json(output / 'proposals.json', {'points': points,
            'scope': 'Integer optimization of the discrete additive proxy; exact only when solver.optimal is true. Validation and measured E2E still required.'})
    write_json(RUN / 'configs/design.json', design)


if __name__ == '__main__':
    main()
