"""Fit fixed conditional models; select a model on policy holdout before validation."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import REPO, RUN, write_json
from daso.modeling.conditional_quality import fit_conditional_quality
import daso.modeling.quality as quality

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, rows):
    with (RUN / 'results' / name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert not (RUN / 'configs/model_choice.json').exists()
    assert not list((RUN / 'results/validation').glob('*/*.json'))
    design = read(RUN / 'configs/design.json')
    for name, digest in design['source_sha256'].items():
        assert sha(Path(name)) == digest, name
    registry = read(RUN / 'configs/policy_registry.json')
    assert sha(RUN / 'configs/policy_registry.json') == design['calibration_registry_sha256']
    data = {}
    source_hashes = {}
    for pid in registry:
        path = RUN / f'calibration/results/validation/{pid}/prefill_decode.json'
        result = read(path)
        assert result['data_role'] == 'calibration' and result['data_sha256'] == design['calibration']['sha256']
        assert len(result['blocks']) == 64
        data[pid] = torch.tensor([r['avg_nll'] for r in result['blocks']], dtype=torch.float64)
        source_hashes[pid] = sha(path)
    reference, anchor = data['uniform_dense_bf16'], data['uniform_sq_int8']
    singles = torch.stack([data[p] for p in design['singles']])
    paired = torch.stack([data[p] for p in design['paired']])
    simple = fit_conditional_quality(reference, anchor, singles)
    pair_model = fit_conditional_quality(reference, anchor, singles,
        pairs=tuple(tuple(p) for p in design['pairs']), pair_nll=paired)
    for label, model in [('conditional_additive', simple), ('conditional_pair', pair_model)]:
        write_json(RUN / f'results/model_{label}.json', {'baseline_delta': model.baseline_delta,
            'group_deltas': model.group_deltas.tolist(), 'pairs': model.pairs,
            'pair_corrections': model.pair_corrections.tolist()})

    actions = read(RUN / 'reference/actions.json')
    methods, modules = actions['methods'], actions['names']
    quality.METHODS = tuple(methods)
    errors = {(r['module'], r['phase'], r['method']): float(r['output_rel_mse'])
              for r in csv.DictReader((RUN / 'reference/local_errors.csv').open())}

    def features(pid):
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        x = torch.zeros((2, len(methods), 4, 4), dtype=torch.float64)
        for name, pair in mapping.items():
            bucket = int(name.split('.')[2]) * 4 // 36
            kind = quality.FUSED_TYPES.index(name.rsplit('.', 1)[-1])
            for i, phase in enumerate(['prefill', 'decode']):
                method = pair[phase + '_method']
                if method != 'dense_bf16':
                    x[i, methods.index(method), bucket, kind] += errors[name, phase, method]
        return x

    training = design['fit_policies']
    x = torch.stack([features(p) for p in training])
    y = torch.tensor([float((data[p] - reference).mean()) for p in training], dtype=torch.float64)
    original = quality.QualityCalibrator(**design['original_refit']).fit(x, y, torch.ones(len(training), dtype=torch.bool))
    original.save(RUN / 'results/model_refit_original.json')
    original_anchor = float(original.predict(features('uniform_sq_int8')))
    original_deltas = torch.tensor([float(original.predict(features(p))) - original_anchor
                                    for p in design['singles']], dtype=torch.float64)
    q = torch.tensor(actions['quality'], dtype=torch.float64)
    old_deltas = torch.tensor([sum(float(q[144 + modules.index(n), methods.index('w4a16_ours')]
        - q[144 + modules.index(n), methods.index('sq_int8')]) for n in group) for group in design['groups']], dtype=torch.float64)

    codes = np.arange(65536, dtype=np.uint32)
    masks = torch.from_numpy(((codes[:, None] >> np.arange(16)) & 1).astype(np.float64))
    predictions = {
        'old_anchored': (anchor - reference).mean() + masks @ old_deltas,
        'refit_original': original_anchor + masks @ original_deltas,
        'conditional_additive': simple.predict(masks), 'conditional_pair': pair_model.predict(masks)}
    holdout_rows, metrics = [], []
    for name, predicted in predictions.items():
        actual = np.array([float((data[p] - reference).mean()) for p in design['holdouts']])
        pred = np.array([float(predicted[registry[p]['mask']]) for p in design['holdouts']])
        difference = pred - actual
        metrics.append({'model': name, 'policies': len(actual), 'mae': float(abs(difference).mean()),
            'rmse': float(np.sqrt(np.mean(difference**2))), 'maximum_error': float(abs(difference).max()),
            'false_feasible_e001': int(((pred <= .01) & (actual > .01)).sum())})
        for pid, a, b in zip(design['holdouts'], actual, pred, strict=True):
            holdout_rows.append({'model': name, 'policy': pid, 'w4_modules': int(registry[pid]['mask']).bit_count() * 9,
                                 'actual_delta_nll': a, 'predicted_delta_nll': b, 'error': b - a})
    primary = min([m for m in metrics if m['model'] != 'old_anchored'], key=lambda m: (m['mae'], m['model']))['model']
    save('policy_holdout_predictions.csv', holdout_rows)
    save('policy_holdout_metrics.csv', metrics)
    write_json(RUN / 'configs/model_choice.json', {'primary_model': primary, 'metrics': metrics,
        'chosen_at_unix': time.time(), 'rule': design['primary_model_rule'], 'calibration_sha256': source_hashes,
        'design_sha256': sha(RUN / 'configs/design.json'), 'scope': 'Policy holdout on calibration windows; no final validation or confirmation used.'})

    profile = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
               for r in csv.DictReader((RUN / 'reference/profile.csv').open())}
    base_cost = sum(profile[n.rsplit('.', 1)[-1], 'sq_int8', 16384]
                    + 63 * profile[n.rsplit('.', 1)[-1], 'sq_int8', 8] for n in modules)
    saved = np.array([sum(63 * (profile[n.rsplit('.', 1)[-1], 'sq_int8', 8]
                    - profile[n.rsplit('.', 1)[-1], 'w4a16_ours', 8]) for n in group) for group in design['groups']])
    assert np.all(saved > 0)
    costs = base_cost - masks.numpy() @ saved
    mask_policy = {item['mask']: pid for pid, item in registry.items() if item['mask'] is not None}
    template = read(RUN / 'configs/policies/uniform_sq_int8.json')
    points, families = [], {}
    for name, predicted in predictions.items():
        values = predicted.numpy()
        families[name] = []
        for target in design['proposal_targets']:
            feasible = np.flatnonzero(values <= target + 1e-12)
            if not len(feasible):
                points.append({'model': name, 'target': target, 'feasible': False})
                continue
            minimum = float(costs[feasible].min())
            tied = feasible[costs[feasible] <= minimum + 1e-9]
            code = int(tied[np.lexsort((tied, values[tied]))[0]])
            assert values[code] <= target + 1e-12 and costs[code] <= minimum + 1e-9
            if code not in mask_policy:
                pid = f'group_{code:04x}'
                changed = {n for g, members in enumerate(design['groups']) if code & (1 << g) for n in members}
                mapping = {n: {'prefill_method': 'sq_int8', 'decode_method': 'w4a16_ours' if n in changed else 'sq_int8'} for n in modules}
                path = RUN / f'configs/policies/{pid}.json'
                write_json(path, {**template, 'policy_id': pid, 'method_map': mapping})
                registry[pid] = {'mask': code, 'role': 'optimizer_proposal', 'policy_sha256': sha(path),
                    'signature': hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()}
                mask_policy[code] = pid
            pid = mask_policy[code]
            families[name].append(pid)
            points.append({'model': name, 'target': target, 'feasible': True, 'policy': pid, 'mask': code,
                'w4_modules': code.bit_count() * 9, 'predicted_delta_nll': float(values[code]),
                'predicted_linear_ms': float(costs[code]), 'enumerated_masks': 65536,
                'cost_tie_tolerance_ms': 1e-9})
    pids = sorted(set(design['anchors']) | {p for ps in families.values() for p in ps})
    assert len(pids) <= 20
    write_json(RUN / 'configs/policy_registry.json', registry)
    write_json(RUN / 'configs/proposal_pool.json', {'policies': pids, 'families': families, 'points': points,
        'primary_model': primary, 'model_choice_sha256': sha(RUN / 'configs/model_choice.json'),
        'registry_sha256': sha(RUN / 'configs/policy_registry.json'), 'scope': 'Only generated masks plus shared anchors enter final validation; not all calibration policies.'})
    write_json(RUN / 'configs/validation_jobs.json', {'jobs': [{'policy': p} for p in pids]})
    write_json(RUN / 'results/fit_complete.json', {'complete': True, 'primary_model': primary,
        'validation_policies': len(pids), 'enumerated_masks_per_model': 65536})
    print('Fitted', metrics, 'primary=', primary, 'validation policies=', len(pids), flush=True)


if __name__ == '__main__':
    main()
