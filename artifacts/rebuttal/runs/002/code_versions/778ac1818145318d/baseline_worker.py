from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

from common import MODEL, RUN, VLLM, WRAPPER, write_json


TEMP = Path('/dev/shm/daso_rebuttal_002_20261025')


def invoke(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    # Every invocation gets its own immutable log, including failed attempts.
    if log.exists():
        log = log.with_name(log.stem + '_' + str(time.time_ns()) + log.suffix)
    files = sorted((RUN / 'code').glob('*.py'))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = RUN / 'code_versions' / version
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for path in files:
            shutil.copy2(path, snapshot / path.name)
    write_json(log.with_suffix('.command.json'), {'command': command, 'code_version': version,
        'code_hashes': hashes, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    with log.open('w') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {log}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policies', required=True)
    parser.add_argument('--actions', default='export,speed,nll,tasks')
    args = parser.parse_args()
    actions = args.actions.split(',')
    backend = ['--vllm-root', str(VLLM), '--cutlass-wrapper-path', str(WRAPPER)]
    status = []
    for policy_id in args.policies.split(','):
        lock_path=RUN/'results/locks'/f'{policy_id}.lock'
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        policy_lock=lock_path.open('a')
        fcntl.flock(policy_lock,fcntl.LOCK_EX)
        policy = RUN / 'configs/policies' / f'{policy_id}.json'
        checkpoint = TEMP / 'checkpoints' / policy_id
        for action in actions:
            started = time.perf_counter()
            print(policy_id, action, 'started', flush=True)
            try:
                if action == 'export':
                    if (checkpoint / 'phase_hetero_export_provenance.json').is_file() and (checkpoint/'model.safetensors').is_file():
                        continue
                    if checkpoint.exists():
                        checkpoint.rename(checkpoint.with_name(checkpoint.name+'_previous_'+str(time.time_ns())))
                    command = [sys.executable, str(RUN / 'code/export_model.py'),
                        '--model-path', str(MODEL), '--policy-json', str(policy),
                        '--output-dir', str(checkpoint), '--cutlass-wrapper-path', str(WRAPPER)]
                    content = policy.read_text()
                    for method in ['sparse_bf16', 'sparse_nvfp4']:
                        if method in content:
                            command.extend(['--canonical-' + method.replace('_', '-') + '-state',
                                str(TEMP / 'canonical' / method / 'model.pt')])
                else:
                    deadline = time.monotonic() + 3600
                    while not (checkpoint / 'phase_hetero_export_provenance.json').is_file():
                        if time.monotonic() > deadline:
                            raise TimeoutError(f'checkpoint not ready: {checkpoint}')
                        time.sleep(2)
                    command = [sys.executable, str(RUN / 'code/run_workflow.py')]
                    if action == 'speed':
                        target = RUN / 'results/speed_fixed_kv3gb' / policy_id
                        if (target / 'summary.csv').is_file():
                            continue
                        command += ['speed', '--checkpoint', str(checkpoint), '--scenario', 'prefill_decode',
                            '--output-dir', str(target), *backend, '--warmup-iters', '2', '--iters', '5',
                            '--reuse-llm', '--gpu-memory-utilization', '.85']
                    elif action == 'nll':
                        target = RUN / 'results/nll' / f'{policy_id}_test_prefill.json'
                        if target.is_file():
                            continue
                        command += ['prefill_nll', '--checkpoint', str(checkpoint), '--tokenizer', str(MODEL),
                            '--samples', str(RUN / 'data/evaluation_test_100_2048.pt'),
                            '--output', str(target), '--label', policy_id, '--phase-hetero',
                            '--policy-json', str(policy), '--blocks', '100', *backend]
                    elif action == 'tasks':
                        task_root=RUN/'results/tasks'/policy_id
                        if all((task_root/task/'full/result.json').is_file() for task in
                               ['wikitext','winogrande','arc_easy','arc_challenge','mmlu']):
                            continue
                        manifest = RUN / 'configs' / f'manifest_{policy_id}.json'
                        write_json(manifest, {'model_path': str(MODEL), 'policies': [{'label': policy_id, 'kind': 'ours',
                            'checkpoint': str(checkpoint), 'policy_json': str(policy),
                            'policy_sha256': hashlib.sha256(policy.read_bytes()).hexdigest()}]})
                        command += ['tasks', '--manifest', str(manifest), '--policy', policy_id,
                            '--output-root', str(RUN / 'results/tasks'), *backend, '--audit']
                    elif action == 'decode_nll':
                        target = RUN / 'results/nll' / f'{policy_id}_test_decode.json'
                        if target.is_file():
                            continue
                        command += ['decode_nll', '--model', str(checkpoint), '--tokenizer', str(MODEL),
                            '--samples', str(RUN / 'data/evaluation_test_100_2112.pt'),
                            '--output', str(target), '--label', policy_id, '--phase-hetero',
                            '--blocks','100','--input-tokens','2048','--output-tokens','64',
                            '--batch-size','8','--gpu-memory-utilization','.70',*backend]
                    elif action == 'generation':
                        command = [sys.executable,str(RUN/'code/generation_worker.py'),'--policy',policy_id]
                    else:
                        raise ValueError(action)
                invoke(command, RUN / 'logs' / f'{policy_id}_{action}.log')
                status.append({'policy': policy_id, 'action': action, 'ok': True,
                               'elapsed_seconds': time.perf_counter() - started})
            except Exception as exc:
                status.append({'policy': policy_id, 'action': action, 'ok': False,
                               'error': repr(exc), 'elapsed_seconds': time.perf_counter() - started})
                print('FAILED', status[-1], flush=True)
            write_json(RUN / f'results/worker_status_{os.getpid()}.json', status)
            print(policy_id, action, 'finished', flush=True)
        policy_lock.close()
    if any(not entry['ok'] for entry in status):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
