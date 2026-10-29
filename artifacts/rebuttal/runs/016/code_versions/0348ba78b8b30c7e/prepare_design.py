"""Freeze training recipes and new nonoverlapping confirmation windows."""
import hashlib
import json
import shutil

import numpy as np
import torch

from common import RUN, write_json

METHODS = ['dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4',
           'w4a16_ours', 'sq_int8', 'fp8', 'int8']
KINDS = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (RUN / 'configs/frozen_design.json').exists()
    template = read(RUN / 'configs/policies/uniform_dense_bf16.json')
    names = list(template['method_map'])
    old_design = read(RUN.parent / '005/configs/design.json')
    datasets = {}
    for split, count, seed in [('validation', 32, 14901671), ('test', 64, 14901673)]:
        path = RUN.parent / f'001/data/wikitext_{split}_tokens.pt'
        tokens = torch.load(path, weights_only=True)
        excluded = set(old_design['data'][split]['starts'])
        available = [s for s in range(0, tokens.numel() - 2112 + 1, 2112) if s not in excluded]
        rng = np.random.default_rng(seed)
        rng.shuffle(available)
        starts = available[:count]
        assert len(starts) == count and not set(starts) & excluded
        label = 'confirmation' if split == 'test' else 'validation'
        destination = RUN / f'data/{label}_{count}_2112.pt'
        torch.save(torch.stack([tokens[s:s+2112] for s in starts]), destination)
        datasets[label] = {'file': str(destination), 'blocks': count, 'starts': starts,
            'excluded_old_starts': sorted(excluded), 'source': str(path), 'source_sha256': sha(path),
            'sha256': sha(destination), 'seed': seed,
            'scope': 'Disjoint from the historical 005/015 64-window set. The benchmark has prior aggregate evaluations; not a wholly unseen corpus.'}
    training = []
    for scene, seed in [('prefill_only', 14901681), ('prefill_decode', 14901683)]:
        old = read(RUN.parent / f'002/configs/calibration_{scene}.json')
        manifest = []
        for entry in old:
            pid = entry['policy_id']
            shutil.copy2(RUN.parent / f'002/configs/policies/{pid}.json', RUN / f'configs/policies/{pid}.json')
            source = RUN.parent / f'002/results/calibration/{scene}/{pid}.json'
            target = RUN / f'results/training/{scene}/{pid}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
            manifest.append({**entry, 'origin': '002', 'source_sha256': sha(source)})
        rng = np.random.default_rng(seed)
        for index in range(24):
            pid = f'cal16_{scene}_{index:02d}'
            mapping = {name: {'prefill_method': 'dense_bf16', 'decode_method': 'dense_bf16'} for name in names}
            if index == 0:
                mapping = {name: {'prefill_method': 'int8', 'decode_method': 'int8'} for name in names}
            elif index <= 16:
                bucket, kind = divmod(index - 1, 4)
                for name, pair in mapping.items():
                    if int(name.split('.')[2]) * 4 // 36 == bucket and name.endswith(KINDS[kind]):
                        pair['prefill_method'] = 'int8' if scene == 'prefill_only' or index % 3 != 0 else 'dense_bf16'
                        pair['decode_method'] = pair['prefill_method'] if scene == 'prefill_only' else ('int8' if index % 3 != 1 else 'dense_bf16')
            else:
                rate = [.1, .25, .4, .55, .7, .85, 1.][index - 17]
                for pair in mapping.values():
                    for phase in ['prefill', 'decode']:
                        pair[phase + '_method'] = ('int8' if rng.random() < rate else
                            str(rng.choice(['dense_bf16', 'dense_nvfp4', 'w4a16_ours', 'sq_int8', 'fp8'])))
                    if scene == 'prefill_only':
                        pair['decode_method'] = pair['prefill_method']
            write_json(RUN / f'configs/policies/{pid}.json', {**template, 'policy_id': pid, 'scenario': scene, 'method_map': mapping})
            item = {'policy_id': pid, 'scenario': scene, 'split': 'train' if index < 18 else 'holdout',
                    'policy_kind': 'new plain INT8 calibration', 'origin': '016'}
            training.append(item)
            manifest.append(item)
        write_json(RUN / f'configs/calibration_{scene}.json', manifest)
    generator = torch.Generator().manual_seed(149016881)
    write_json(RUN / 'configs/calibration_speed_inputs.json', torch.randint(100, 150000, (8, 2048), generator=generator).tolist())
    write_json(RUN / 'configs/training_jobs.json', training)
    write_json(RUN / 'configs/frozen_design.json', {'methods': METHODS, 'names': names, 'datasets': datasets,
        'new_training_labels_per_scenario': 24, 'new_train': 18, 'new_holdout': 6,
        'quality_fit': {'steps': 4000, 'learning_rate': .03, 'regularization': .05,
                        'architecture': 'Unchanged positive global+phase+method+bucket+type calibrator'},
        'latency_fractions': [.95, .80, .65], 'quality_budgets': [.01, .04],
        'matched_methods': ['dense_bf16', 'int8', 'dense_nvfp4'],
        'matched_pd_modes': ['same_operator', 'phase_pair'], 'hessian_primary': 'signed',
        'hessian_central_robustness': ['clip_zero', 'absolute'],
        'validation_guard': {'z': 2., 'margin': .005}, 'speed_calibration_iterations': 5,
        'speed_test_rounds': 3, 'speed_test_iterations': 8, 'warmups': 3,
        'profile_sha256': sha(RUN / 'reference/graph_action_profile.csv'),
        'bf16_calibration_sha256': sha(RUN / 'reference/calibration_uniform_dense_bf16.json'),
        'runtime_sources_sha256': sha(RUN / 'reference/runtime_sources.json'),
        'training_protocol': '100 raw train blocks; prefill B1/S2048/O1; teacher-forced P+D B8/S2048/O64, eager to match inherited labels; 3GiB BF16 KV.',
        'nll_evaluation_protocol': 'Validation32 and confirmation64, B8/S2048/O1 or64, native decode Graph,6GiB BF16 KV; all batches full.',
        'constraints': 'Identical candidate action pairs and common latency cap for the two allocators; original infeasibility retained without silent clamp.',
        'selection': 'Freeze all allocation points before validation/calibration, freeze selected full-eight-action maps before opening confirmation labels. No test-based change of Hessian convention, phase mode or budgets.'})
    print('Frozen 48 new calibration labels, validation32 and confirmation64.', flush=True)


if __name__ == '__main__':
    main()
