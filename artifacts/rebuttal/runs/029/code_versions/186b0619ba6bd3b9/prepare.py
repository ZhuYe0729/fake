"""Freeze conditional decoder calibration, policy holdouts and fresh data."""
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
SOURCE = RUN.parent / '028'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not (RUN / 'configs/design.json').exists()
    cleanup = REPO / 'artifacts/rebuttal/cleanup/20261027_pre_daso_optimization/completed.json'
    assert read(cleanup)['complete'] and read(cleanup)['deleted_files'] == 45
    for d in ['code', 'configs/policies', 'data/checkpoints', 'results', 'logs', 'reference', 'calibration/configs']:
        (RUN / d).mkdir(parents=True, exist_ok=True)
    names = ['common.py', 'eightbit.py', 'runtime_extensions.py', 'arena_runtime.py', 'graph_bindings.py',
             'int4_marlin.py', 'telemetry.py', 'export_model.py', 'generate_tasks.py', 'score_nll.py', 'benchmark.py',
             'resources.py', 'selection.py']
    for name in names:
        shutil.copy2(SOURCE / 'code' / name, RUN / 'code' / name)
    for d in ['compat', 'csrc']:
        shutil.copytree(SOURCE / 'code' / d, RUN / 'code' / d)
    for name in ['activation_scales.pt', 'generation_cnn_dm_1000.jsonl', 'generation_dsum.jsonl', 'generation_IWSLT.jsonl']:
        (RUN / 'data' / name).symlink_to((SOURCE / 'data' / name).resolve())
    for a, b in [('reference/actions.json', 'reference/actions.json'), ('reference/profile.csv', 'reference/profile.csv')]:
        shutil.copy2(SOURCE / a, RUN / b)
    shutil.copy2(RUN.parent / '016/data/local_errors_extended.csv', RUN / 'reference/local_errors.csv')
    frozen = read(RUN / 'reference/actions.json')
    modules = frozen['names']
    kinds = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']
    groups = [[n for n in modules if int(n.split('.')[2]) * 4 // 36 == bucket and n.endswith('.' + kind)]
              for bucket in range(4) for kind in kinds]
    assert len(groups) == 16 and all(len(g) == 9 for g in groups)
    pairs = [(4 * b + k, 4 * b + k + 1) for b in range(4) for k in [0, 2]]
    holdouts = [0x1111, 0x2222, 0x4444, 0x8888, 0x3333, 0xCCCC, 0x5555, 0xAAAA]
    template = read(SOURCE / 'configs/policies/uniform_sq_int8.json')
    registry = {}

    def add(pid, mapping, role, mask=None):
        path = RUN / f'configs/policies/{pid}.json'
        write(path, {**template, 'policy_id': pid, 'scenario': 'prefill_decode', 'method_map': mapping})
        registry[pid] = {'policy_sha256': sha(path), 'signature': hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode()).hexdigest(), 'role': role, 'mask': mask}
        return pid

    anchors = ['uniform_dense_bf16', 'uniform_int8', 'uniform_sq_int8', 'phase_sq_int8_w4a16_ours']
    for pid in anchors:
        add(pid, read(SOURCE / f'configs/policies/{pid}.json')['method_map'], 'anchor',
            0 if pid == 'uniform_sq_int8' else 65535 if pid == 'phase_sq_int8_w4a16_ours' else None)

    def add_mask(mask, role):
        changed = {n for g, members in enumerate(groups) if mask & (1 << g) for n in members}
        return add(f'group_{mask:04x}', {n: {'prefill_method': 'sq_int8',
            'decode_method': 'w4a16_ours' if n in changed else 'sq_int8'} for n in modules}, role, mask)

    singles = [add_mask(1 << g, 'single_group_fit') for g in range(16)]
    paired = [add_mask((1 << a) | (1 << b), 'pair_fit') for a, b in pairs]
    heldout = [add_mask(mask, 'policy_holdout') for mask in holdouts]
    assert len(registry) == 36 and len({p['signature'] for p in registry.values()}) == 36
    write(RUN / 'configs/policy_registry.json', registry)
    write(RUN / 'configs/calibration_jobs.json', {'jobs': [{'policy': p} for p in registry]})

    original = RUN.parent / '001/data'
    calibration = torch.load(original / 'calibration_128_2112.pt', weights_only=True)[:64].clone()
    path = RUN / 'data/calibration_64_2112.pt'
    torch.save(calibration, path)
    cal = {'file': str(path), 'blocks': 64, 'sha256': sha(path), 'source': str(original / 'calibration_128_2112.pt')}
    write(RUN / 'calibration/configs/frozen_design.json', {'datasets': {'validation': cal}})
    tokens = torch.load(original / 'wikitext_train_tokens.pt', weights_only=True)
    excluded = read(original / 'provenance.json')['calibration']['starts']
    excluded += read(SOURCE / 'configs/design.json')['all_new_starts']
    available = [i for i in range(0, len(tokens) - 2112 + 1, 2112)
                 if all(i + 2112 <= j or j + 2112 <= i for j in excluded)]
    starts = random.Random(15402931).sample(available, 272)
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
    paths = [REPO / 'daso/src/daso/modeling/conditional_quality.py', REPO / 'daso/src/daso/modeling/quality.py',
             SOURCE / 'code/driver.py', RUN / 'reference/actions.json', RUN / 'reference/profile.csv', RUN / 'reference/local_errors.csv']
    design = {'created_at_unix': time.time(), 'scope': 'Fixed SQ-INT8 prefill,16 groups of9 modules; internal corpus/policy holdouts; not full144-module global search.',
        'anchors': anchors, 'groups': groups, 'pairs': pairs, 'singles': singles, 'paired': paired,
        'holdouts': heldout, 'calibration': cal, 'datasets': datasets, 'all_new_starts': starts,
        'excluded_starts': excluded, 'token_source': str(original / 'wikitext_train_tokens.pt'),
        'budgets': [.01, .04], 'proposal_targets': [0., .005, .01, .04],
        'models': ['old_anchored', 'refit_original', 'conditional_additive', 'conditional_pair'],
        'primary_model_rule': 'Lowest eight-policy calibration-holdout MAE among refit_original/conditional_additive/conditional_pair; ties by model name; chosen before final validation.',
        'original_refit': {'steps': 4000, 'learning_rate': .03, 'regularization': .05},
        'guard': 'paired mean+2SE+.005, BF16 zero; calibration E2E primary; cycle separately reported',
        'fit_policies': ['uniform_dense_bf16', 'uniform_sq_int8', *singles, *paired],
        'source_sha256': {str(p): sha(p) for p in paths},
        'runtime_source_sha256': {name: sha(SOURCE / 'code' / name) for name in names},
        'protected_files': {str(p): sha(p) for p in [SOURCE / 'results/final_audit.json', RUN.parent / '027/results/final_audit.json',
            RUN.parent / '015/results/manifest.json', RUN.parent / '016/results/manifest.json']},
        'calibration_registry_sha256': sha(RUN / 'configs/policy_registry.json'), 'cleanup_sha256': sha(cleanup)}
    write(RUN / 'configs/design.json', design)
    (RUN / 'README.md').write_text('# Qwen3条件增量质量校准\n\n清理153之后的新DASO诊断：固定SQ-INT8 prefill，36个预定校准map，16个decode组和8个pair交互。仅用policy holdout确定主模型，再生成至多20个提案/anchor，128新validation与128新confirmation。状态见results/worker.json，完成后见REPORT.md。\n')
    print('Frozen36 calibration maps,16groups/8pairs,8policy holdouts; fresh128validation+128confirmation.', flush=True)


if __name__ == '__main__':
    main()
