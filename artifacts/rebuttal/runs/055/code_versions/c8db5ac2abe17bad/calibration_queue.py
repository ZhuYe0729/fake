"""Run a fixed shard of fresh 32B labels; preserve all results and export identity."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import MODEL, RUN, WRAPPER, write_json


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def invoke(command, label):
    write_json(RUN / f'logs/{label}.command.json', {'argv': [str(x) for x in command],
        'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'time_ns': time.time_ns()})
    with (RUN / f'logs/{label}.log').open('w') as log:
        subprocess.run([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT, check=True)


def snapshot_code():
    sources = [path for path in sorted((RUN / 'code').rglob('*'))
               if path.is_file() and '__pycache__' not in path.parts]
    identities = {str(path.relative_to(RUN / 'code')): digest(path) for path in sources}
    signature = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    target = RUN / 'code_versions' / signature[:16]
    if not target.exists():
        target.mkdir(parents=True)
        for source in sources:
            destination = target / source.relative_to(RUN / 'code')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        write_json(target / 'manifest.json', identities)
    return str(target), signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=72)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(args.gpu)
    manifest = json.loads((RUN / 'configs/calibration_manifest.json').read_text())
    design = json.loads((RUN / 'configs/design.json').read_text())
    with (RUN / f'gpu{args.gpu}.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        for entry in manifest['entries'][args.begin:args.end:args.stride]:
            policy = entry['policy_id']
            target = RUN / f'results/calibration_complete/{policy}.json'
            if target.exists():
                assert json.loads(target.read_text())['complete']
                continue
            policy_path = RUN / f'configs/policies/{policy}.json'
            assert digest(policy_path) == entry['policy_sha256']
            code_snapshot, code_signature = snapshot_code()
            label = f'calibration_{policy}_g{args.gpu}_{time.time_ns()}'
            if policy == 'uniform_dense_bf16':
                checkpoint = RUN / 'exports/uniform_dense_bf16'
                temporary = False
            else:
                checkpoint = Path('/dev/shm/daso_rebuttal_055') / label
                temporary = True
                with (RUN / 'export.lock').open('a') as export_lock:
                    fcntl.flock(export_lock, fcntl.LOCK_EX)
                    assert shutil.disk_usage('/dev/shm').free > 140 * 1024**3
                    command = [sys.executable, RUN / 'code/export_model.py', '--model-path', MODEL,
                        '--policy-json', policy_path, '--output-dir', checkpoint, '--cutlass-wrapper-path', WRAPPER]
                    for method in ['sparse_bf16', 'sparse_nvfp4']:
                        command += ['--canonical-' + method.replace('_', '-') + '-state',
                                    RUN / f'data/canonical/{method}/model.pt']
                    invoke(command, label + '_export')
            evidence = RUN / f'results/export_identity/{label}'
            evidence.mkdir(parents=True)
            for source in checkpoint.glob('*.json'):
                shutil.copy2(source, evidence / source.name)
            binary = checkpoint / 'model.safetensors'
            identity = {'policy': policy, 'policy_sha256': entry['policy_sha256'],
                'binary_sha256': digest(binary), 'binary_bytes': binary.stat().st_size,
                'checkpoint': str(checkpoint), 'temporary': temporary,
                'code_snapshot': code_snapshot, 'code_signature': code_signature,
                'source_code_sha256': {str(path.relative_to(RUN)): digest(path)
                    for path in sorted((RUN / 'code').rglob('*')) if path.is_file() and '__pycache__' not in path.parts}}
            write_json(evidence / 'identity.json', identity)
            invoke([sys.executable, RUN / 'code/experiment.py', '--checkpoint', checkpoint,
                '--policy', policy, '--mode', 'quality', '--quality-split', 'calibration',
                '--tier', '1', '--label', label], label)
            results = []
            for case in design['workloads']:
                path = RUN / f"results/quality/calibration/{policy}/{case['label']}.json"
                result = json.loads(path.read_text())
                assert result['complete'] and len(result['blocks']) == 100
                results.append({'path': str(path), 'sha256': digest(path)})
            write_json(target, {'complete': True, 'policy': policy, 'gpu': args.gpu,
                'export_identity': str(evidence / 'identity.json'), 'quality': results})
            if temporary:
                binary.unlink()
                write_json(evidence / 'temporary_weight_cleanup.json', {
                    'path': str(binary), 'bytes': identity['binary_bytes'],
                    'sha256': identity['binary_sha256'], 'reason': 'Only this run-owned reproducible weight binary, after successful label collection.',
                    'preserved': 'All source, configuration, canonical states, captures, logs, and measured results.'})
            print('completed', policy, 'GPU', args.gpu, flush=True)


if __name__ == '__main__':
    main()
