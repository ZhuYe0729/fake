from __future__ import annotations

import hashlib
import json
import platform
import subprocess

import torch
import transformers

from common import MODEL, REPO, RUN, VLLM, WRAPPER, write_json


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def main() -> None:
    config = json.loads((MODEL / 'config.json').read_text())
    names = [f'model.layers.{index}.{suffix}'
             for index in range(config['num_hidden_layers'])
             for suffix in ['self_attn.qkv_proj', 'self_attn.o_proj', 'mlp.gate_up_proj', 'mlp.down_proj']]
    (RUN / 'configs/policies').mkdir(exist_ok=True)
    for method in ['dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4',
                   'w4a16_ours', 'int8', 'int4']:
        policy = {'policy_id': 'uniform_' + method, 'scenario': 'prefill_decode',
                  'default_prefill_method': method, 'default_decode_method': method,
                  'modules_to_not_convert': ['lm_head'],
                  'method_map': {name: {'prefill_method': method, 'decode_method': method}
                                 for name in names}}
        write_json(RUN / f'configs/policies/uniform_{method}.json', policy)
    backend_files = {}
    for relative in ['artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py',
                     'artifacts/dev/012_phase_hetero_linear/phase_hetero_mytest.py',
                     'vllm/vllm/model_executor/models/qwen3.py']:
        path = VLLM / relative
        if path.is_file():
            backend_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(RUN / 'configs/environment.json', {
        'model': str(MODEL), 'model_config': config, 'repo_commit': revision(REPO),
        'vllm_root': str(VLLM), 'vllm_commit': revision(VLLM),
        'wrapper_root': str(WRAPPER), 'wrapper_commit': revision(WRAPPER),
        'backend_files': backend_files, 'python': platform.python_version(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'temporary_root': '/dev/shm/daso_rebuttal_001_20261025',
        'workloads': {'prefill_only': [8, 2048, 1], 'prefill_decode': [8, 2048, 64]},
        'generation_chat_template': 'native Qwen3, enable_thinking=False',
        'int8_backend': 'torch._int_mm; installed vLLM CUTLASS rejects SM120',
        'cleanup': 'No existing model, code, log, CSV or document has been removed.'})
    print('Wrote uniform policies and environment provenance', flush=True)


if __name__ == '__main__':
    main()
