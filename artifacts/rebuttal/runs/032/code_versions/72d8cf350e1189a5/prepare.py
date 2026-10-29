"""Freeze budget/phase-weight sensitivity before downstream evaluation."""
import csv
import hashlib
import itertools
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
sys.path.insert(0, str(REPO / 'daso/src'))
from daso.modeling.phase_quality import combine_phase_quality
from exact_solver import solve

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    assert not (RUN / 'configs/design.json').exists()
    frozen = read(RUN / 'reference/actions.json')
    names, methods = frozen['names'], frozen['methods']
    q = np.array(frozen['quality'])
    pairs = list(itertools.product(range(8), repeat=2))
    assert len(names) == 144 and len(methods) == 8
    raw = combine_phase_quality(q[:144], q[144:]).numpy().reshape(144, 64)
    profile = {(r['module_type'], r['method'], int(r['m'])): float(r['latency_ms'])
               for r in csv.DictReader((RUN / 'reference/profile.csv').open())}
    costs = np.array([[profile[n.rsplit('.', 1)[-1], methods[p], 16384] +
                       63 * profile[n.rsplit('.', 1)[-1], methods[d], 8] for p, d in pairs] for n in names])
    registry, signatures, points = {}, {}, []
    template = read(RUN.parent / '031/configs/policies/uniform_dense_bf16.json')

    def add(pid, mapping, role):
        signature = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        if signature in signatures:
            return signatures[signature]
        path = RUN / f'configs/policies/{pid}.json'
        write(path, {**template, 'policy_id': pid, 'method_map': mapping})
        registry[pid] = {'signature': signature, 'role': role, 'policy_sha256': sha(path)}
        signatures[signature] = pid
        return pid

    anchors = []
    for rid, pid in [('031', 'uniform_dense_bf16'), ('031', 'uniform_int8'), ('031', 'uniform_sq_int8'),
                     ('031', 'phase_sq_int8_w4a16_ours'), ('029', 'group_7fff')]:
        anchors.append(add(pid, read(RUN.parent / rid / f'configs/policies/{pid}.json')['method_map'], 'control'))
    settings = [(1., e) for e in [.01, .02, .03, .04]] + [(w, .03) for w in [1.5, 2.]]
    for weight, budget in settings:
        weighted = combine_phase_quality(q[:144], q[144:], prefill_weight=weight).numpy().reshape(144, 64)
        first = solve(weighted, costs, budget)
        assert first['optimal']
        second = solve(weighted, costs, first['latency_ms'] + 1e-8, inverse=True)
        assert second['optimal'] and second['quality'] <= budget + 1e-9
        mapping = {n: {'prefill_method': methods[pairs[i][0]], 'decode_method': methods[pairs[i][1]]}
                   for n, i in zip(names, second['indices'])}
        label = f'w{weight:g}_e{budget:g}'
        pid = add(label, mapping, 'predeclared full-space sweep')
        if weight == 1. and budget == .04:
            assert mapping == read(RUN.parent / '031/configs/policies/old_anchored_0.04.json')['method_map']
        points.append({'label': label, 'policy': pid, 'prefill_weight': weight, 'budget': budget,
            'weighted_score': second['quality'], 'original_prediction': float(raw[np.arange(144), second['indices']].sum()),
            'linear_ms': second['latency_ms'], 'latency_solver': first, 'quality_tie_solver': second})
        print(label, pid, second['quality'], second['latency_ms'], flush=True)
    write(RUN / 'configs/policy_registry.json', registry)
    write(RUN / 'configs/points.json', points)
    write(RUN / 'configs/selected_policies.json', {'selected': [{'policy': p} for p in registry],
        'frozen_at_unix': time.time(), 'rule': 'All predeclared maps evaluated; no downstream score based selection.'})
    source = RUN.parent / '001/data/evaluation_test_100_2112.pt'
    path = RUN / 'data/benchmark_test_100_2112.pt'
    data = torch.load(source, weights_only=True)[:100].clone()
    assert tuple(data.shape) == (100, 2112)
    torch.save(data, path)
    dataset = {'file': str(path), 'blocks': 100, 'sha256': sha(path), 'source': str(source), 'source_sha256': sha(source),
        'scope': 'Previously evaluated fixed WikiText test benchmark; not new blind confirmation.'}
    write(RUN / 'configs/frozen_design.json', {'datasets': {'confirmation': dataset}})
    for kind, seed in [('speed_calibration', 15703217), ('speed_formal', 15703219)]:
        indices = random.Random(seed).sample(range(100), 8)
        write(RUN / f'configs/{kind}.json', [data[i, :2048].tolist() for i in indices])
    write(RUN / 'configs/evaluation_jobs.json', {'jobs': [{'policy': p} for p in registry]})
    generation, reused = [], []
    same_code = ['generate_tasks.py', 'eightbit.py', 'runtime_extensions.py', 'arena_runtime.py', 'graph_bindings.py']
    for pid in registry:
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        matches = []
        for rid, prefix in [('027', 'full'), ('028', ''), ('029', '')]:
            root = RUN.parent / rid
            assert all(sha(RUN / 'code' / n) == sha(root / 'code' / n) for n in same_code)
            for p in (root / 'configs/policies').glob('*.json'):
                if read(p)['method_map'] == mapping:
                    matches.append((root / prefix / f'results/generation/{p.stem}', rid))
        for task in ['cnn_dm_1000', 'dsum', 'IWSLT']:
            match = next(((root / task / 'metrics_0000_end.json', rid) for root, rid in matches
                          if (root / task / 'metrics_0000_end.json').exists()), None)
            if match:
                source, rid = match
                reused.append({'policy': pid, 'dataset': task, 'source': str(source), 'source_run': rid,
                    'sha256': sha(source), 'metric': read(source), 'same_map': True})
            else:
                generation.append({'policy': pid, 'dataset': task})
    write(RUN / 'configs/generation_jobs.json', {'jobs': generation, 'reused': reused})
    # MC evaluates prompt likelihoods only; identical prefill maps share one
    # fresh MC evaluation, independently of their unused decode representation.
    representatives, mc_source = {}, {}
    for pid in registry:
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        digest = hashlib.sha256(json.dumps({n: v['prefill_method'] for n, v in mapping.items()}, sort_keys=True).encode()).hexdigest()
        mc_source[pid] = representatives.setdefault(digest, pid)
    write(RUN / 'configs/mc_jobs.json', {'jobs': [{'policy': p} for p in representatives.values()],
        'prefill_equivalent_source': mc_source, 'tasks': ['arc_challenge', 'arc_easy', 'winogrande', 'mmlu']})
    core = REPO / 'daso/src/daso/modeling/phase_quality.py'
    shutil.copy2(core, RUN / 'reference/core/phase_quality.py')
    source_paths = [core, RUN/'reference/actions.json', RUN/'reference/profile.csv',
        RUN.parent/'031/results/final_audit.json', RUN.parent/'016/code/run_workflow.py',
        REPO/'daso/workflows/paper/runtime/evaluate_prefill_tasks.py']
    design = {'created_at_unix': time.time(), 'anchors': anchors, 'settings': settings,
        'source_sha256': {str(p.resolve()): sha(p) for p in source_paths},
        'canonical_sha256': read(RUN.parent/'031/results/final_audit.json')['canonical_source_sha256'],
        'dataset': dataset, 'registry_sha256': sha(RUN/'configs/policy_registry.json'),
        'points_sha256': sha(RUN/'configs/points.json'), 'selected_sha256': sha(RUN/'configs/selected_policies.json'),
        'speed_formal_sha256': sha(RUN/'configs/speed_formal.json'),
        'scope': 'Budget/phase-weight sensitivity; same full space, no adaptive task-score selection. Weighted score is not calibrated delta NLL.'}
    write(RUN / 'configs/design.json', design)
    write(RUN / 'results/design_checks.json', {'verified': True, 'modules': 144, 'actions_per_module': 64,
        'settings': 6, 'unique_maps': len(registry), 'generation_new_jobs': len(generation),
        'generation_reused_jobs': len(reused), 'mc_unique_prefill_maps': len(representatives)})
    print('Frozen', read(RUN/'results/design_checks.json'), flush=True)


if __name__ == '__main__':
    main()
