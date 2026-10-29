"""Sequential isolated diagnostic processes; retain exact commands and failures."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
TEMP = Path('/dev/shm/daso_rebuttal_014_20261026')
MODEL = Path('/root/wja/data/models/Qwen/Qwen3-8B')
WRAPPER = RUN.parents[3] / 'fake/kernels/cutlass/cutlass_wrapper'


def invoke(command, label):
    command = ['numactl', '--physcpubind=0-7', '--membind=0', *map(str, command)]
    stamp = f'{label}_{time.time_ns()}'
    log = RUN / f'logs/{stamp}.log'
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUN / 'code').glob('*.py')}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = RUN / 'code_versions' / version
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in (RUN / 'code').glob('*.py'):
        if not (snapshot / p.name).exists():
            shutil.copy2(p, snapshot / p.name)
    log.with_suffix('.command.json').write_text(json.dumps({'command': command, 'code_version': version,
        'code_hashes': hashes, 'gpu': 0}, indent=2) + '\n')
    start = time.perf_counter()
    with log.open('w') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
    log.with_suffix('.completion.json').write_text(json.dumps({'returncode': result.returncode,
        'seconds': time.perf_counter() - start}, indent=2) + '\n')
    print(label, result.returncode, str(log), flush=True)
    if result.returncode:
        raise RuntimeError(str(log))


def main():
    policies = sys.argv[1:] or ['uniform_dense_bf16', 'uniform_w4a16_ours', 'uniform_marlin_int4']
    for policy in policies:
        checkpoint = (Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_dense_bf16')
                      if policy == 'uniform_dense_bf16' else TEMP / 'checkpoints' / policy)
        if not (checkpoint / 'model.safetensors').exists():
            assert not checkpoint.exists(), 'Preserve existing export before rebuilding.'
            invoke([sys.executable, RUN / 'code/export_model.py', '--model-path', MODEL,
                    '--policy-json', RUN / f'configs/policies/{policy}.json',
                    '--output-dir', checkpoint, '--cutlass-wrapper-path', WRAPPER], policy + '_export')
        invoke([sys.executable, RUN / 'code/diagnose_decode.py', '--checkpoint', checkpoint,
                '--policy', policy, '--label', policy], policy)


if __name__ == '__main__':
    main()
