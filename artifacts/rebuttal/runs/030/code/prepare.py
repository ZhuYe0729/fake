"""Reuse029 calibration; forecast the final guard before generating proposals."""
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
SOURCE = RUN.parent / '029'
sys.path.insert(0, str(REPO / 'daso/src'))
from daso.modeling.quality_constraint import forecast_quality_guard

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not (RUN / 'configs/design.json').exists()
    source = read(SOURCE / 'configs/design.json')
    assert read(SOURCE / 'configs/model_choice.json')['primary_model'] == 'conditional_additive'
    for d in ['code', 'configs/policies', 'data/checkpoints', 'logs', 'results', 'reference']:
        (RUN / d).mkdir(parents=True, exist_ok=True)
    files = ['common.py', 'eightbit.py', 'runtime_extensions.py', 'arena_runtime.py', 'graph_bindings.py',
        'int4_marlin.py', 'telemetry.py', 'export_model.py', 'generate_tasks.py', 'score_nll.py',
        'resources.py', 'selection.py', 'benchmark.py', 'score.py', 'ops.py']
    for name in files:
        shutil.copy2(SOURCE / 'code' / name, RUN / 'code' / name)
    for d in ['compat', 'csrc']:
        shutil.copytree(SOURCE / 'code' / d, RUN / 'code' / d)
    # Retain the exact orchestration, adding formal timing for the mean-only
    # selections as controls. The primary pool is cumulative, never weakened.
    driver = (SOURCE / 'code/driver.py').read_text()
    before = "        if not (RUN / 'results/formal_complete.json').exists():\n            b.formal()"
    after = "        if not (RUN / 'results/formal_complete.json').exists():\n            comparisons = read(RUN / 'configs/model_selections.json')['choices']\n            b.CONTROLS = sorted(set(b.CONTROLS) | {p['policy'] for p in comparisons})\n            b.formal()"
    assert before in driver
    (RUN / 'code/driver.py').write_text(driver.replace(before, after))
    for name in ['activation_scales.pt', 'generation_cnn_dm_1000.jsonl', 'generation_dsum.jsonl', 'generation_IWSLT.jsonl']:
        (RUN / 'data' / name).symlink_to((SOURCE / 'data' / name).resolve())
    shutil.copy2(SOURCE / 'reference/profile.csv', RUN / 'reference/profile.csv')
    source_registry = read(SOURCE / 'configs/policy_registry.json')
    dependencies = {}

    def nll(pid):
        path = SOURCE / f'calibration/results/validation/{pid}/prefill_decode.json'
        dependencies[str(path)] = sha(path)
        return torch.tensor([b['avg_nll'] for b in read(path)['blocks']], dtype=torch.float64)

    reference, base = nll('uniform_dense_bf16'), nll('uniform_sq_int8')
    differences = torch.stack([nll(p) - base for p in source['singles']])
    codes = np.arange(65536, dtype=np.uint32)
    masks = torch.from_numpy(((codes[:, None] >> np.arange(16)) & 1).astype(np.float64))
    initial = forecast_quality_guard(base - reference, differences, masks, validation_windows=128)
    residuals = [float((nll(p) - reference).mean()) - float(initial.mean[source_registry[p]['mask']])
                 for p in source['holdouts']]
    residual = max(0., max(residuals))
    forecast = forecast_quality_guard(base - reference, differences, masks,
                                     validation_windows=128, residual_margin=residual)
    profile = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
               for r in csv.DictReader((RUN / 'reference/profile.csv').open())}
    modules = [n for g in source['groups'] for n in g]
    base_cost = sum(profile[n.rsplit('.', 1)[-1], 'sq_int8', 16384]
                    + 63 * profile[n.rsplit('.', 1)[-1], 'sq_int8', 8] for n in modules)
    saved = np.array([sum(63 * (profile[n.rsplit('.', 1)[-1], 'sq_int8', 8]
                    - profile[n.rsplit('.', 1)[-1], 'w4a16_ours', 8]) for n in g) for g in source['groups']])
    costs = base_cost - masks.numpy() @ saved
    registry, mask_policy = {}, {}
    template = read(SOURCE / 'configs/policies/uniform_sq_int8.json')

    def inherit(pid):
        old = SOURCE / f'configs/policies/{pid}.json'
        target = RUN / f'configs/policies/{pid}.json'
        shutil.copy2(old, target)
        registry[pid] = {**source_registry[pid], 'source_policy_sha256': sha(old), 'policy_sha256': sha(target)}
        if registry[pid]['mask'] is not None:
            mask_policy[registry[pid]['mask']] = pid

    anchors = source['anchors']
    old_pool = read(SOURCE / 'configs/proposal_pool.json')
    mean_points = sorted(set(old_pool['families']['conditional_additive']))
    for pid in sorted(set(anchors) | set(mean_points)):
        inherit(pid)
    points, guard_policies = [], []
    for target in [.005, .01, .02, .04]:
        values = forecast.guarded.numpy()
        feasible = np.flatnonzero(values <= target + 1e-12)
        if not len(feasible):
            points.append({'target': target, 'feasible': False})
            continue
        minimum = float(costs[feasible].min())
        tied = feasible[costs[feasible] <= minimum + 1e-9]
        code = int(tied[np.lexsort((tied, values[tied]))[0]])
        assert values[code] <= target + 1e-12 and costs[code] <= minimum + 1e-9
        if code not in mask_policy:
            pid = f'group_{code:04x}'
            changed = {n for g, names in enumerate(source['groups']) if code & (1 << g) for n in names}
            mapping = {n: {'prefill_method': 'sq_int8', 'decode_method': 'w4a16_ours' if n in changed else 'sq_int8'} for n in modules}
            target_path = RUN / f'configs/policies/{pid}.json'
            write(target_path, {**template, 'policy_id': pid, 'method_map': mapping})
            registry[pid] = {'mask': code, 'role': 'guard_aware_proposal', 'policy_sha256': sha(target_path),
                'signature': hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()}
            mask_policy[code] = pid
        pid = mask_policy[code]
        guard_policies.append(pid)
        points.append({'target': target, 'feasible': True, 'policy': pid, 'mask': code,
            'w4_modules': code.bit_count() * 9, 'predicted_mean': float(forecast.mean[code]),
            'predicted_se': float(forecast.standard_error[code]), 'predicted_guard': float(forecast.guarded[code]),
            'linear_ms': float(costs[code]), 'enumerated_masks': 65536})
    assert len(registry) <= 10
    write(RUN / 'configs/policy_registry.json', registry)
    write(RUN / 'configs/model_choice.json', {'primary_model': 'guard_aware', 'chosen_at_unix': time.time(),
        'rule': 'Predeclared guard-aware cumulative proposal rule; not chosen by030 validation/confirmation.'})
    write(RUN / 'configs/proposal_pool.json', {'policies': sorted(registry),
        'families': {'mean_only': mean_points, 'guard_aware': sorted(set(mean_points) | set(guard_policies))},
        'primary_model': 'guard_aware', 'points': points,
        'model_choice_sha256': sha(RUN / 'configs/model_choice.json'),
        'registry_sha256': sha(RUN / 'configs/policy_registry.json')})
    write(RUN / 'configs/validation_jobs.json', {'jobs': [{'policy': p} for p in sorted(registry)]})
    write(RUN / 'results/guard_forecast.json', {'residual_margin': residual,
        'policy_holdout_residuals': dict(zip(source['holdouts'], residuals, strict=True)),
        'formula': 'mean +2*sqrt(paired_covariance_variance/128)+.005+max_positive_calibration_policy_residual',
        'scope': 'Forecast of128-window validation statistic, not formal population guarantee.'})

    original = RUN.parent / '001/data'
    tokens = torch.load(original / 'wikitext_train_tokens.pt', weights_only=True)
    excluded = [*source['excluded_starts'], *source['all_new_starts']]
    available = [i for i in range(0, len(tokens) - 2112 + 1, 2112)
                 if all(i + 2112 <= j or j + 2112 <= i for j in excluded)]
    starts = random.Random(15503041).sample(available, 272)
    datasets = {}
    for split, first in [('validation', 0), ('confirmation', 128)]:
        values = starts[first:first + 128]
        path = RUN / f'data/{split}_128_2112.pt'
        torch.save(torch.stack([tokens[i:i + 2112] for i in values]), path)
        datasets[split] = {'file': str(path), 'blocks': 128, 'starts': values, 'sha256': sha(path)}
    for name, first in [('speed_calibration', 256), ('speed_formal', 264)]:
        path = RUN / f'configs/{name}.json'
        values = starts[first:first + 8]
        write(path, [tokens[i:i + 2048].tolist() for i in values])
        datasets[name] = {'file': str(path), 'starts': values, 'sha256': sha(path)}
    write(RUN / 'configs/frozen_design.json', {'datasets': datasets})
    paths = [REPO / 'daso/src/daso/modeling/quality_constraint.py', REPO / 'daso/src/daso/modeling/conditional_quality.py',
        SOURCE / 'configs/design.json', SOURCE / 'configs/model_choice.json', SOURCE / 'configs/proposal_pool.json',
        SOURCE / 'configs/selected_policies.json', RUN / 'reference/profile.csv']
    design = {'created_at_unix': time.time(), 'anchors': anchors, 'groups': source['groups'],
        'models': ['mean_only', 'guard_aware'], 'budgets': [.01, .04], 'guard_targets': [.005, .01, .02, .04],
        'source_sha256': {str(p): sha(p) for p in paths}, 'calibration_reuse_sha256': dependencies,
        'datasets': datasets, 'all_new_starts': starts, 'excluded_starts': excluded,
        'source_run': '029', 'new_calibration_jobs': 0,
        'scope': 'Prospective continuation after029 development validation, without029 confirmation feedback; new030 final windows exclude all prior windows.'}
    write(RUN / 'configs/design.json', design)
    for name in ['calibration_complete', 'fit_complete']:
        write(RUN / f'results/{name}.json', {'complete': True, 'reused_from': '029', 'new_calibration_jobs': 0})
    assert len(set(starts)) == 272 and all(b >= a + 2112 for a, b in zip(sorted(starts), sorted(starts)[1:]))
    assert all(a + 2112 <= b or b + 2112 <= a for a in starts for b in excluded)
    write(RUN / 'results/design_checks.json', {'verified': True, 'policies': len(registry),
        'enumerated_masks': 65536, 'fresh_token_disjoint_windows': 272, 'calibration_reused': True})
    (RUN / 'README.md').write_text('# Qwen3 guard-aware提案\n\n复用029训练校准，用配对协方差+校准holdout残差预测验证保护值；均值提案全部保留。全新128validation+128confirmation，排除之前全部窗口。状态见results/worker.json；完成后见REPORT.md。\n')
    print('Prepared', len(registry), 'maps; residual margin', residual, 'guard points', points, flush=True)


if __name__ == '__main__':
    main()
