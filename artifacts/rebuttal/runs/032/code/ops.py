"""Reuse the audited runner with a new registry and short per-job GPU leases."""
import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import MODEL, REPO, RUN, WRAPPER, write_json

read = lambda p: json.loads(p.read_text())
source = RUN.parent / '028/code/driver.py'
spec = importlib.util.spec_from_file_location('runner028_adapter', source)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
release = b.cleanup


def state(stage, **details):
    write_json(RUN / 'results/worker.json', {'status': 'running', 'stage': stage, **details})


def checkpoint(pid, gpu):
    entry = read(RUN / 'configs/policy_registry.json')[pid]
    policy = RUN / f'configs/policies/{pid}.json'
    assert b.sha(policy) == entry['policy_sha256']
    mapping = read(policy)['method_map']
    sources = list((RUN.parent / '027/data/checkpoints').glob('*'))
    sources.append(Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_dense_bf16'))
    for root in sources:
        if (root / 'model.safetensors').exists() and read(root / 'config.json')['quantization_config']['method_map'] == mapping:
            write_json(RUN / f'results/checkpoint_reuse/{pid}.json', {'source': str(root),
                'effective_map_equal': True, 'config_sha256': b.sha(root / 'config.json')})
            return root
    model = RUN / 'data/checkpoints' / pid
    if not (model / 'model.safetensors').exists():
        with (b.ROOT / 'export.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if not (model / 'model.safetensors').exists():
                assert shutil.disk_usage(RUN).free > 64 * 1024**3
                if model.exists():
                    model.rename(model.with_name(pid + '_previous_' + str(time.time_ns())))
                command = [sys.executable, RUN / 'code/export_model.py', '--model-path', MODEL,
                    '--policy-json', policy, '--output-dir', model, '--cutlass-wrapper-path', WRAPPER]
                for method in ['sparse_bf16', 'sparse_nvfp4']:
                    if any(method in pair.values() for pair in mapping.values()):
                        canonical = REPO / 'artifacts/rebuttal/night151/cache/daso_rebuttal_001_20261025/canonical' / method / 'model.pt'
                        assert canonical.exists(), canonical
                        command += ['--canonical-' + method.replace('_', '-') + '-state', canonical]
                b.invoke(command, 'export_' + pid, gpu)
    assert read(model / 'config.json')['quantization_config']['method_map'] == mapping
    return model


# Proposal checkpoints stay immutable through validation/calibration/formal/tasks.
# Calibration perturbations are released explicitly after their one scoring job.
b.checkpoint = checkpoint
b.cleanup = lambda *args: None
b.CONTROLS = read(RUN / 'configs/design.json')['anchors']


def worker(stage):
    from resources import lease
    jobs = read(RUN / f'configs/{stage}_jobs.json')['jobs']
    for job in jobs:
        pid = job['policy']
        label = pid + ('_' + job['dataset'] if stage == 'generation' else '')
        marker = RUN / f'results/completed/{stage}/{label}.json'
        if marker.exists():
            continue
        lock = RUN / f'results/locks/{stage}_{label}.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if marker.exists():
                continue
            with lease() as gpu:
                model = checkpoint(pid, gpu)
                if stage == 'mc':
                    manifest = RUN / f'configs/manifest_{pid}.json'
                    write_json(manifest, {'model_path': str(MODEL), 'policies': [{'label': pid, 'kind': 'ours',
                        'checkpoint': str(model), 'policy_json': str(RUN / f'configs/policies/{pid}.json'),
                        'policy_sha256': b.sha(RUN / f'configs/policies/{pid}.json')}]})
                    argv = [sys.executable, RUN / 'code/run_workflow.py', 'tasks', '--manifest', manifest,
                        '--policy', pid, '--output-root', RUN / 'results/tasks', '--vllm-root', REPO.parent / 'test/vllm',
                        '--cutlass-wrapper-path', WRAPPER, '--audit']
                elif stage == 'generation':
                    argv = [sys.executable, RUN / 'code/generate_tasks.py', '--checkpoint', model,
                            '--policy', pid, '--dataset', job['dataset']]
                else:
                    argv = [sys.executable, RUN / 'code/score.py', '--checkpoint', model,
                            '--policy', pid, '--role', stage]
                b.invoke(argv, stage + '_' + label, gpu)
                write_json(marker, {'complete': True, 'gpu': gpu, **job})
                if stage == 'calibration':
                    release(model, stage + '_' + label)


def quality_phase(stage):
    total = len(read(RUN / f'configs/{stage}_jobs.json')['jobs'])
    children = []
    for i in range(3):
        log = (RUN / f'logs/{stage}_worker{i}_{time.time_ns()}.log').open('w')
        child = subprocess.Popen([sys.executable, __file__, '--worker', stage], stdout=log, stderr=subprocess.STDOUT)
        children.append((child, log))
    while any(p.poll() is None for p, _ in children):
        state(stage, complete=len(list((RUN / f'results/completed/{stage}').glob('*.json'))), total=total,
              workers=[{'pid': p.pid, 'returncode': p.poll()} for p, _ in children])
        time.sleep(10)
    codes = [p.returncode for p, _ in children]
    for _, log in children:
        log.close()
    assert codes == [0, 0, 0], codes
    assert len(list((RUN / f'results/completed/{stage}').glob('*.json'))) == total
    write_json(RUN / f'results/{stage}_complete.json', {'complete': True, 'jobs': total})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--run-stage')
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.worker)
        else:
            assert args.run_stage
            quality_phase(args.run_stage)
    except Exception as error:
        write_json(RUN / f'results/failure_{os.getpid()}_{time.time_ns()}.json', {'error': repr(error)})
        raise
