"""Freeze the existing main-table rows and their missing evaluation jobs."""
import hashlib
import json
import shutil
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
TABLE = REPO / 'artifacts/rebuttal/responses/reviewer2/qwen3_table2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not (RUN / 'configs/design.json').exists()
    for name in ['common.py', 'eightbit.py', 'runtime_extensions.py', 'export_model.py', 'int4_marlin.py']:
        shutil.copy2(RUN.parent / '011/code' / name, RUN / 'code' / name)
    shutil.copytree(RUN.parent / '011/code/compat', RUN / 'code/compat')
    for name in ['run_workflow.py', 'generate_tasks.py']:
        shutil.copy2(RUN.parent / '010/code' / name, RUN / 'code' / name)
    shutil.copy2(RUN.parent / '012/code/telemetry.py', RUN / 'code/telemetry.py')
    text = (RUN.parent / '012/code/benchmark.py').read_text()
    text = text.replace("for output in design['outputs']:", "for output in design['policies'][args.policy]['outputs']:")
    text = text.replace('args=parser.parse_args()', "parser.add_argument('--record-id');args=parser.parse_args()")
    text = text.replace('/{args.policy}.json', '/{args.record_id or args.policy}.json')
    text = text.replace("ids=torch.randint(100,30000,(8,2048),generator=torch.Generator().manual_seed(spec['seed']))",
        "input_path=run/'configs/speed_inputs.json';input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest();assert input_sha==design['speed_inputs_sha256']\n    ids=torch.tensor(json.loads(input_path.read_text()),dtype=torch.long)")
    text = text.replace("'round':args.round,'split':spec['split']", "'input_sha256':input_sha,'round':args.round,'split':spec['split']")
    (RUN / 'code/benchmark.py').write_text(text)
    snapshot = RUN / 'reference/table_before'
    snapshot.mkdir()
    for path in TABLE.iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
    for name in ['activation_scales.pt']:
        (RUN / 'data' / name).symlink_to((RUN.parent / '011/data' / name).resolve())
    for name in ['likelihood_datasets', 'likelihood_auxiliary_datasets',
                 'generation_cnn_dm_1000.jsonl', 'generation_dsum.jsonl', 'generation_IWSLT.jsonl']:
        source = (RUN.parent / '010/data' / name).resolve()
        assert source.exists(), source
        (RUN / 'data' / name).symlink_to(source, target_is_directory=source.is_dir())
    rows = json.loads((TABLE / 'provenance.json').read_text())['rows']
    signatures, policies, frozen_rows, jobs = {}, {}, [], []
    (RUN / 'configs/policies').mkdir()
    for row in rows:
        entry = {'method': row['method']}
        for scenario in ['prefill_only', 'prefill_decode']:
            old = row['policies']
            origin, original_id = old.get(scenario + '_speed') or old[scenario + '_quality']
            source = RUN.parent / origin / f'configs/policies/{original_id}.json'
            policy = json.loads(source.read_text())
            signature = hashlib.sha256(json.dumps(policy['method_map'], sort_keys=True).encode()).hexdigest()
            pid = signatures.setdefault(signature, original_id)
            if pid not in policies:
                policy['policy_id'] = pid
                write(RUN / f'configs/policies/{pid}.json', policy)
                policies[pid] = {'signature': signature, 'source': str(source), 'source_sha256': digest(source), 'outputs': []}
            output = 1 if scenario == 'prefill_only' else 64
            if output not in policies[pid]['outputs']:
                policies[pid]['outputs'].append(output)
            quality = old.get(scenario + '_quality')
            if quality is None:
                quality = ['013', pid]
                kind = 'tasks' if scenario == 'prefill_only' else 'generation'
                if not any(j['policy'] == pid and j['kind'] == kind for j in jobs):
                    jobs.append({'policy': pid, 'kind': kind})
            entry[scenario] = {'policy': pid, 'quality': quality}
        frozen_rows.append(entry)
    assert len(frozen_rows) == 17 and len(jobs) == 5
    order = [p for p in policies if p != 'uniform_dense_bf16']
    assert sum(len(p['outputs']) for p in policies.values()) == 32
    orders = [['uniform_dense_bf16', *o] for o in [order, list(reversed(order)), order[7:] + order[:7]]]
    import torch
    inputs = RUN / 'configs/speed_inputs.json'
    inputs.write_text(json.dumps(torch.randint(100, 30000, (8, 2048), generator=torch.Generator().manual_seed(144983)).tolist()) + '\n')
    write(RUN / 'configs/design.json', {'registered_time_ns': time.time_ns(), 'rows': frozen_rows, 'policies': policies,
        'speed_inputs_sha256': digest(inputs),
        'quality_jobs': jobs, 'quality_gpus': [1, 2, 4, 5, 7], 'speed_gpu': 0,
        'rounds': [{'round': i, 'split': 'repeat', 'seed': 144983, 'order': o} for i, o in enumerate(orders)],
        'warmups': 3, 'iterations': 8, 'batch': 8, 'input': 2048, 'kv_bytes': 6 * 1024**3,
        'speed_placement': {'node': 0, 'cpus': '0-7'},
        'selection': 'Existing 17 rows and their exact policy maps; no new search or task-score selection.',
        'speed': 'Three complete campaigns, per-iteration first-token observation before FINAL_ONLY; restore recorded separately.'})
    write(RUN / 'reference/protected_manifests.json', {f'{i:03}': digest(RUN.parent / f'{i:03}/results/manifest.json') for i in range(1, 13)})
    print(json.dumps({'rows': len(rows), 'unique_maps': len(policies), 'policy_scenarios': 32, 'quality_jobs': jobs}))


if __name__ == '__main__':
    main()
