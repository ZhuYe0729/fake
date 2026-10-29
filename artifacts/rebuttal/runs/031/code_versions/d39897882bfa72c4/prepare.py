"""Freeze a full-module, full-phase quality repair without new test feedback."""
import csv
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
sys.path.insert(0, str(REPO / 'daso/src'))
from daso.modeling.residual_quality import fit_residual_quality
from exact_solver import solve

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    assert not (RUN / 'configs/design.json').exists()
    source = RUN.parent / '016'
    extra = RUN.parent / '029'
    frozen = read(RUN / 'reference/actions.json')
    names, methods = frozen['names'], frozen['methods']
    pairs = list(itertools.product(range(8), repeat=2))
    assert len(names) == 144 and len(methods) == 8 and methods[0] == 'dense_bf16'
    q = np.array(frozen['quality'])
    prior = np.array([[q[i, p] + q[144 + i, d] for p, d in pairs] for i in range(144)])
    # Counts are features, not tied decisions: every module has 64 free actions.
    features = torch.zeros(144, 64, 320, dtype=torch.float64)
    kinds = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']
    for i, name in enumerate(names):
        bucket = int(name.split('.')[2]) * 4 // 36
        kind = kinds.index(name.rsplit('.', 1)[-1])
        for j, (p, d) in enumerate(pairs):
            for phase, method in enumerate([p, d]):
                if method:
                    features[i, j, phase * 128 + method * 16 + bucket * 4 + kind] = 1 / 9
            if p or d:
                features[i, j, 256 + p * 8 + d] = 1 / 144

    dependencies, records = {}, {}

    def load(path):
        dependencies[str(path)] = sha(path)
        return read(path)

    def nll(path):
        value = load(path)
        return np.array([b['avg_nll'] for b in value['blocks'][:64]])

    baseline_old = nll(source / 'results/training/prefill_decode/cal_prefill_decode_p00.json')
    baseline_new = nll(extra / 'calibration/results/validation/uniform_dense_bf16/prefill_decode.json')
    assert np.max(abs(baseline_old - baseline_new)) < 3e-7
    assert torch.equal(torch.load(source / 'data/calibration_128_2112.pt', weights_only=True)[:64],
                       torch.load(extra / 'data/calibration_64_2112.pt', weights_only=True))

    def add(pid, policy, result, baseline, holdout, origin):
        mapping = policy['method_map']
        signature = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        indices = [pairs.index(tuple(methods.index(mapping[n][phase + '_method']) for phase in ['prefill', 'decode'])) for n in names]
        delta = nll(result) - baseline
        row = {'policy': pid, 'indices': indices, 'delta': delta.tolist(), 'holdout': holdout,
               'origin': origin, 'signature': signature}
        if signature in records:
            holdout = holdout or records[signature]['holdout']
            if origin == '016':
                records[signature]['holdout'] = holdout
                return
            row['holdout'] = holdout
        records[signature] = row

    for entry in load(source / 'configs/calibration_prefill_decode.json'):
        pid = entry['policy_id']
        add(pid, load(source / f'configs/policies/{pid}.json'),
            source / f'results/training/prefill_decode/{pid}.json', baseline_old,
            entry['split'] != 'train', '016')
    design_extra = load(extra / 'configs/design.json')
    for pid in design_extra['fit_policies'] + design_extra['holdouts']:
        add(pid, load(extra / f'configs/policies/{pid}.json'),
            extra / f'calibration/results/validation/{pid}/prefill_decode.json', baseline_new,
            pid in design_extra['holdouts'], '029')
    records = list(records.values())
    selections = torch.tensor([r['indices'] for r in records])
    x = features[torch.arange(144)[None], selections].sum(1)
    y = torch.tensor([np.mean(r['delta']) for r in records], dtype=torch.float64)
    base = torch.tensor([prior[np.arange(144), r['indices']].sum() for r in records])
    train = torch.tensor([not r['holdout'] for r in records])
    metrics, fits = [], {}
    for regularization in [.001, .01, .1, 1., 10.]:
        fit = fit_residual_quality(x[train], y[train], base[train], regularization=regularization)
        pred = fit.predict(x, base)
        fits[regularization] = fit
        metrics.append({'regularization': regularization, 'train_mae': float(abs(pred[train] - y[train]).mean()),
                        'holdout_mae': float(abs(pred[~train] - y[~train]).mean())})
    chosen = min(metrics, key=lambda r: (r['holdout_mae'], -r['regularization']))['regularization']
    fit = fits[chosen]
    corrected = prior + (features @ fit.coefficients).numpy()
    assert np.all(corrected[:, 0] == 0.)
    direct = fit.predict(x, base).numpy()
    summed = np.array([corrected[np.arange(144), r['indices']].sum() for r in records])
    assert np.max(abs(direct - summed)) < 1e-10
    write(RUN / 'results/calibration_labels.json', records)
    write(RUN / 'results/model_fit.json', {'metrics': metrics, 'chosen_regularization': chosen,
        'train_policies': int(train.sum()), 'holdout_policies': int((~train).sum()),
        'old_holdout_mae': float(abs(base[~train] - y[~train]).mean()),
        'new_holdout_mae': float(abs(torch.tensor(direct)[~train] - y[~train]).mean()),
        'bf16_protocol_max_difference': float(np.max(abs(baseline_old - baseline_new))),
        'coefficients': fit.coefficients.tolist(), 'old_bias_removed': frozen['bias'],
        'note': 'Both priors anchored to BF16=0. Old labels eager,029 labels Graph; same tokens,own paired BF16. Policy holdouts select ridge only; no final windows used.'})
    profile = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
               for r in csv.DictReader((RUN / 'reference/profile.csv').open())}
    costs = np.array([[profile[n.rsplit('.', 1)[-1], methods[p], 16384] +
                       63 * profile[n.rsplit('.', 1)[-1], methods[d], 8] for p, d in pairs] for n in names])
    template = read(RUN.parent / '030/configs/policies/uniform_dense_bf16.json')
    registry, signatures = {}, {}

    def policy(pid, mapping, role):
        signature = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        if signature in signatures:
            return signatures[signature]
        path = RUN / f'configs/policies/{pid}.json'
        write(path, {**template, 'policy_id': pid, 'method_map': mapping})
        registry[pid] = {'signature': signature, 'role': role, 'policy_sha256': sha(path)}
        signatures[signature] = pid
        return pid

    anchors = []
    for pid in ['uniform_dense_bf16', 'uniform_int8', 'uniform_sq_int8', 'phase_sq_int8_w4a16_ours', 'group_3ee5']:
        anchors.append(policy(pid, read(RUN.parent / f'030/configs/policies/{pid}.json')['method_map'], 'common anchor'))
    families, points = {}, []
    for model, quality in [('old_anchored', prior), ('signed_residual', corrected)]:
        families[model] = []
        for target in [-.02, 0., .01, .04]:
            if quality.min(1).sum() > target + 1e-10:
                points.append({'model': model, 'target': target, 'feasible': False})
                continue
            result = solve(quality, costs, target)
            assert result['optimal'], result
            mapping = {n: {'prefill_method': methods[pairs[j][0]], 'decode_method': methods[pairs[j][1]]}
                       for n, j in zip(names, result['indices'], strict=True)}
            pid = policy(f'{model}_{target:g}', mapping, 'full-space automatic proposal')
            families[model].append(pid)
            points.append({'model': model, 'target': target, 'feasible': True, 'policy': pid, 'solver': result})
            print(model, target, pid, result['quality'], result['latency_ms'], flush=True)
    assert len(registry) <= 13
    write(RUN / 'configs/policy_registry.json', registry)
    write(RUN / 'configs/model_choice.json', {'primary_model': 'signed_residual', 'chosen_at_unix': time.time(),
        'rule': 'Predeclared residual experiment; regularization from policy holdout, no final validation.'})
    write(RUN / 'configs/proposal_pool.json', {'policies': sorted(registry), 'families': families, 'points': points,
        'primary_model': 'signed_residual', 'model_choice_sha256': sha(RUN / 'configs/model_choice.json')})
    write(RUN / 'configs/validation_jobs.json', {'jobs': [{'policy': p} for p in registry]})
    previous = read(RUN.parent / '030/configs/design.json')
    excluded = previous['excluded_starts'] + previous['all_new_starts']
    tokens = torch.load(RUN.parent / '001/data/wikitext_train_tokens.pt', weights_only=True)
    available = [i for i in range(0, len(tokens) - 2112 + 1, 2112)
                 if all(i + 2112 <= j or j + 2112 <= i for j in excluded)]
    starts = random.Random(15603117).sample(available, 208)
    datasets = {}
    for split, first, count in [('validation', 0, 128), ('confirmation', 128, 64)]:
        path = RUN / f'data/{split}_{count}_2112.pt'
        values = starts[first:first + count]
        torch.save(torch.stack([tokens[i:i + 2112] for i in values]), path)
        datasets[split] = {'file': str(path), 'blocks': count, 'starts': values, 'sha256': sha(path)}
    for split, first in [('speed_calibration', 192), ('speed_formal', 200)]:
        path = RUN / f'configs/{split}.json'
        values = starts[first:first + 8]
        write(path, [tokens[i:i + 2048].tolist() for i in values])
        datasets[split] = {'file': str(path), 'starts': values, 'sha256': sha(path)}
    write(RUN / 'configs/frozen_design.json', {'datasets': datasets})
    core = REPO / 'daso/src/daso/modeling/residual_quality.py'
    (RUN / 'reference/core/residual_quality.py').write_bytes(core.read_bytes())
    dependencies[str(core)] = sha(core)
    dependencies[str(RUN / 'reference/profile.csv')] = sha(RUN / 'reference/profile.csv')
    dependencies[str(RUN / 'reference/actions.json')] = sha(RUN / 'reference/actions.json')
    design = {'created_at_unix': time.time(), 'scope': '144 independent fused modules,each64 free phase pairs; no fixed prefill/group tying.',
        'models': ['old_anchored', 'signed_residual'], 'budgets': [.01, .04], 'anchors': anchors,
        'datasets': datasets, 'source_sha256': dependencies, 'all_new_starts': starts, 'excluded_starts': excluded,
        'calibration_model_sha256': sha(RUN / 'results/model_fit.json'), 'training_policy_sha256': sha(RUN / 'results/calibration_labels.json'),
        'primary_objective': 'single-batch E2E;restore separately reported; no full generation in rapid study'}
    write(RUN / 'configs/design.json', design)
    assert len(set(starts)) == 208
    write(RUN / 'results/design_checks.json', {'verified': True, 'validation_maps': len(registry),
        'modules': 144, 'actions_per_module': 64, 'tied_module_decisions': False,
        'independent_prefill_decode': True, 'train_holdout_disjoint_by_map': True,
        'policy_action_prediction_max_error': float(np.max(abs(direct - summed))), 'fresh_token_disjoint_windows': 208})
    print('Frozen', len(registry), 'maps; fit', {k:v for k,v in read(RUN/'results/model_fit.json').items() if k!='coefficients'}, flush=True)


if __name__ == '__main__':
    main()
