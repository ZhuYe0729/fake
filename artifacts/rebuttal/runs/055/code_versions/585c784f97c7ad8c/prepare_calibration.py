"""Freeze fresh 32B calibration maps before collecting any strategy labels."""
import hashlib
import json

import numpy as np

from common import RUN, write_json


def main():
    target = RUN / 'configs/calibration_manifest.json'
    assert not target.exists()
    design = json.loads((RUN / 'configs/design.json').read_text())
    methods = ['dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4',
               'w4a16_ours', 'int8', 'sq_int8', 'fp8']
    base = json.loads((RUN / 'configs/policies/uniform_dense_bf16.json').read_text())
    names = list(base['method_map'])
    rng = np.random.default_rng(185551)
    rates = np.geomspace(.01, 1., 64)
    rng.shuffle(rates)
    entries = []
    for index in range(72):
        if index < len(methods):
            policy_id = 'uniform_' + methods[index]
            path = RUN / f'configs/policies/{policy_id}.json'
        else:
            policy_id = f'cal32_p{index:02d}'
            path = RUN / f'configs/policies/{policy_id}.json'
            probabilities = rng.dirichlet(np.ones(len(methods) - 1))
            choices = {}
            for name in names:
                pair = {}
                for phase in ['prefill', 'decode']:
                    pair[phase + '_method'] = (str(rng.choice(methods[1:], p=probabilities))
                        if rng.random() < rates[index - len(methods)] else 'dense_bf16')
                choices[name] = pair
            write_json(path, {**base, 'policy_id': policy_id, 'method_map': choices})
            design['policies'][policy_id] = {'display': policy_id, 'scenario': None,
                'map_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        entries.append({'policy_id': policy_id, 'split': 'train' if index < 54 else 'holdout',
                        'kind': 'uniform' if index < len(methods) else 'heterogeneous',
                        'policy_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(target, {'seed': 185551, 'methods': methods, 'entries': entries,
        'train_policies': 54, 'holdout_policies': 18, 'quality_windows': 100,
        'samples': design['quality_samples']['calibration'],
        'sample_sha256': design['quality_sha256']['calibration'],
        'scope': 'Fresh 32B maps and labels; prompt NLL depends on the effective prefill map. '
                 'Any prompt-label reuse for a tied-phase policy requires a measured equality check.',
        'representative_quality_budgets': [.02, .07],
        'selection': 'Fit on train policies; report held-out policy error; representative closure uses validation only.'})
    write_json(RUN / 'configs/design.json', design)
    write_json(RUN / 'configs/engine_frozen.json', {'capacity_probe': 'results/bf16_vllm_feasibility.json',
        'phase_probe': 'results/measurements/pilot_phase_bf16_validation_retry.json',
        'engine': design['engine'], 'workloads': design['workloads'],
        'quality_blocks': design['quality_blocks'], 'warmups': 5, 'iterations': 10, 'rounds': 3,
        'formal_timing': 'No own preparation/quality jobs on either GPU; same GPU/CPU/NUMA; BF16 brackets.'})


if __name__ == '__main__':
    main()
