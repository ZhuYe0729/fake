"""Generate 32B policies and the initial workload design from model metadata."""
import hashlib
import json

import torch

from common import MODEL, RUN, write_json


def main():
    config = json.loads((MODEL / 'config.json').read_text())
    assert config['num_hidden_layers'] == 64
    methods = ['dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4',
               'w4a16_ours', 'int8', 'sq_int8', 'fp8']
    names = [f'model.layers.{layer}.{suffix}' for layer in range(config['num_hidden_layers'])
             for suffix in ['self_attn.qkv_proj', 'self_attn.o_proj', 'mlp.gate_up_proj', 'mlp.down_proj']]
    policies = {}
    for method in methods:
        policy_id = 'uniform_' + method
        path = RUN / f'configs/policies/{policy_id}.json'
        assert not path.exists()
        policy = {'policy_id': policy_id, 'scenario': 'prefill_decode',
                  'default_prefill_method': method, 'default_decode_method': method,
                  'modules_to_not_convert': ['lm_head'],
                  'method_map': {name: {'prefill_method': method, 'decode_method': method} for name in names}}
        write_json(path, policy)
        policies[policy_id] = {'display': method, 'scenario': None,
                               'map_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    speed = RUN / 'configs/speed_inputs.json'
    validation = torch.load(RUN / 'data/validation_100_2112.pt', weights_only=True)
    write_json(speed, validation[64:72, :2048].tolist())
    samples = {split: str(RUN / 'data' / filename) for split, filename in [
        ('calibration', 'calibration_128_2112.pt'), ('validation', 'validation_100_2112.pt'),
        ('test', 'evaluation_test_100_2112.pt')]}
    workloads = []
    for scenario, output in [('prefill_only', 1), ('prefill_decode', 64)]:
        workloads.append({'label': scenario + '_b8_s2048_o' + str(output), 'tier': 1,
                          'scenario': scenario, 'batch': 8, 'prompt': 2048, 'output': output,
                          'input': str(speed), 'input_sha256': hashlib.sha256(speed.read_bytes()).hexdigest()})
    design = {'status': 'pilot; freeze final engine and representative policies before formal measurements',
              'policies': policies, 'workloads': workloads, 'quality_samples': samples,
              'quality_sha256': {split: hashlib.sha256(open(path, 'rb').read()).hexdigest() for split, path in samples.items()},
              'quality_blocks': 100, 'warmups': 5, 'iterations': 10, 'rounds': 3,
              'engine': {'capture_sizes': [1, 4, 8], 'max_model_len': 2304,
                         'max_num_seqs': 8, 'max_num_batched_tokens': 16384, 'kv_gib': 8},
              'module_count': len(names), 'hidden_layers': 64,
              'quality_selection': 'Only calibration/validation inputs may be used to select representative policies.'}
    write_json(RUN / 'configs/design.json', design)


if __name__ == '__main__':
    main()
