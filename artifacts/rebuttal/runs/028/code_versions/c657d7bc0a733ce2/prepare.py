"""Freeze a bounded decoder repair study before reading any new NLL labels."""
import csv
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch

RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN.parent / '016'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not (RUN / 'configs/design.json').exists()
    for name in ['code', 'configs/policies', 'data/checkpoints', 'logs', 'results', 'reference']:
        (RUN / name).mkdir(parents=True, exist_ok=True)
    runtime = RUN.parent / '027/code'
    names = ['common.py', 'eightbit.py', 'runtime_extensions.py', 'arena_runtime.py',
             'graph_bindings.py', 'int4_marlin.py', 'telemetry.py', 'export_model.py', 'generate_tasks.py']
    for name in names:
        shutil.copy2(runtime / name, RUN / 'code' / name)
    for directory in ['compat', 'csrc']:
        shutil.copytree(runtime / directory, RUN / 'code' / directory)
    for name in ['score_nll.py', 'benchmark.py']:
        shutil.copy2(SOURCE / 'code' / name, RUN / 'code' / name)
    for name in ['activation_scales.pt', 'generation_cnn_dm_1000.jsonl', 'generation_dsum.jsonl', 'generation_IWSLT.jsonl']:
        (RUN / 'data' / name).symlink_to((SOURCE / 'data' / name).resolve())
    shutil.copy2(SOURCE / 'data/actions_prefill_decode.json', RUN / 'reference/actions.json')
    shutil.copy2(SOURCE / 'reference/graph_action_profile.csv', RUN / 'reference/profile.csv')
    actions = read(RUN / 'reference/actions.json')
    modules, methods, quality = actions['names'], actions['methods'], actions['quality']
    assert len(modules) == 144 and len(quality) == 288
    lut = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
           for r in csv.DictReader((RUN / 'reference/profile.csv').open())}
    template = read(SOURCE / 'configs/policies/uniform_dense_bf16.json')
    policies, signatures, rankings = {}, set(), {}

    def add(pid, mapping, family, **extra):
        digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        assert digest not in signatures, pid
        signatures.add(digest)
        policy = {**template, 'policy_id': pid, 'scenario': 'prefill_decode', 'method_map': mapping}
        path = RUN / f'configs/policies/{pid}.json'
        write(path, policy)
        cost = sum(lut[n.rsplit('.', 1)[-1], m['prefill_method'], 16384]
                   + 63 * lut[n.rsplit('.', 1)[-1], m['decode_method'], 8] for n, m in mapping.items())
        predicted = actions['bias'] + sum(quality[i][methods.index(mapping[n]['prefill_method'])]
                      + quality[144 + i][methods.index(mapping[n]['decode_method'])] for i, n in enumerate(modules))
        policies[pid] = {'signature': digest, 'policy_sha256': sha(path), 'family': family,
                         'predicted_linear_ms': cost, 'predicted_delta_nll': predicted, **extra}

    for pre in ['sq_int8', 'int8', 'dense_nvfp4']:
        for dec in ['sq_int8', 'int8', 'w4a16_ours']:
            pid = 'uniform_' + pre if pre == dec else f'phase_{pre}_{dec}'
            add(pid, {n: {'prefill_method': pre, 'decode_method': dec} for n in modules}, 'global anchor')
    for method in ['dense_bf16', 'dense_nvfp4', 'w4a16_ours']:
        add('uniform_' + method, {n: {'prefill_method': method, 'decode_method': method} for n in modules}, 'uniform anchor')
    for base in ['sq_int8', 'int8']:
        rows = []
        for i, name in enumerate(modules):
            kind = name.rsplit('.', 1)[-1]
            saved = 63 * (lut[kind, base, 8] - lut[kind, 'w4a16_ours', 8])
            delta = quality[144 + i][methods.index('w4a16_ours')] - quality[144 + i][methods.index(base)]
            assert saved > 0
            rows.append({'name': name, 'delta_proxy': delta, 'saved_ms': saved, 'ratio': delta / saved})
        rows.sort(key=lambda r: (r['ratio'], r['name']))
        rankings[base] = rows
        for count in [8, 16, 32, 64, 96, 128]:
            changed = {r['name'] for r in rows[:count]}
            add(f'repair_{base}_w4_{count:03d}', {n: {'prefill_method': base,
                'decode_method': 'w4a16_ours' if n in changed else base} for n in modules},
                'baseline-anchored decode repair', base=base, replaced=count)
    assert len(policies) == 24
    write(RUN / 'configs/rankings.json', rankings)

    source = RUN.parent / '001/data/wikitext_train_tokens.pt'
    tokens = torch.load(source, weights_only=True)
    original = read(RUN.parent / '001/data/provenance.json')['calibration']['starts']
    available = [i for i in range(0, len(tokens) - 2112 + 1, 2112)
                 if all(i + 2112 <= j or j + 2112 <= i for j in original)]
    starts = random.Random(15202811).sample(available, 208)
    assert len(set(starts)) == 208
    datasets = {}
    for split, first, count in [('validation', 0, 128), ('confirmation', 128, 64)]:
        values = starts[first:first + count]
        path = RUN / f'data/{split}_{count}_2112.pt'
        torch.save(torch.stack([tokens[i:i + 2112] for i in values]), path)
        datasets[split] = {'file': str(path), 'blocks': count, 'starts': values, 'sha256': sha(path)}
    for name, first in [('speed_calibration', 192), ('speed_formal', 200)]:
        path = RUN / f'configs/{name}.json'
        values = starts[first:first + 8]
        write(path, [tokens[i:i + 2048].tolist() for i in values])
        datasets[name] = {'file': str(path), 'starts': values, 'sha256': sha(path)}
    write(RUN / 'configs/frozen_design.json', {'datasets': datasets})
    design = {'created_at_unix': time.time(), 'policies': policies, 'budgets': [.01, .04],
        'validation_blocks': 128, 'confirmation_blocks': 64, 'guard': 'paired mean + 2SE + .005; BF16 zero',
        'selection': 'Minimum measured calibration E2E; separately declared minimum cycle control; feasible INT8 incumbents retained.',
        'validation_order_sensitivity': [32, 64, 128], 'calibration_iterations': 5, 'formal_iterations': 5,
        'formal_campaigns': 3, 'proposal_seed': 15202811, 'datasets': datasets,
        'token_source': str(source.resolve()), 'token_source_sha256': sha(source),
        'excluded_calibration_starts': original, 'all_new_starts': starts,
        'data_scope': 'Previously unused internal WikiText train windows, token-disjoint from original 128-window quality/activation calibration and each other; not a new corpus or official unseen benchmark split.',
        'quality_model_sha256': sha(RUN / 'reference/actions.json'), 'profile_sha256': sha(RUN / 'reference/profile.csv'),
        'runtime_source_hashes': {n: sha(runtime / n) for n in names},
        'original_manifests': {n: sha(RUN.parent / n / 'results/manifest.json') for n in ['015', '016']},
        'generation_rule': 'Full original three tasks for new primary E2E selections and plain INT8 control; exact same-map 027 Graph results may be reused with signatures. Scores never enter selection.'}
    write(RUN / 'configs/design.json', design)
    (RUN / 'README.md').write_text('# Qwen3 baseline-anchored decode repair\n\n独立选点修正：保留 INT8 incumbent，加入部分 W4A16 decode，冻结24个 map、128窗口 validation 和64窗口 confirmation；沿用原 NLL 保护值与预算。当前状态见 results/worker.json；完成后见 REPORT.md。\n')
    print('Frozen 24 maps, 128 validation + 64 confirmation + 16 speed windows.', flush=True)


if __name__ == '__main__':
    main()
