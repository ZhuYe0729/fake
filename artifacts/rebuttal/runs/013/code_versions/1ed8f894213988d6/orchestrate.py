"""Fill fixed quality jobs, then run all speed repeats on a quiet GPU0."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
TEMP = Path('/dev/shm/daso_rebuttal_013_20261026')
MODEL = Path('/root/wja/data/models/Qwen/Qwen3-8B')
WRAPPER = REPO / 'fake/kernels/cutlass/cutlass_wrapper'
BASELINE = Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_dense_bf16')


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def invoke(argv, label, gpu):
    node = 0 if gpu < 4 else 1
    first = gpu * 8
    command = ['numactl', '--physcpubind=' + f'{first}-{first+7}', '--membind=' + str(node), *map(str, argv)]
    log = RUN / f'logs/{label}_{time.time_ns()}.log'
    files = sorted((RUN / 'code').glob('*.py'))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = RUN / 'code_versions' / version
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in files:
        if not (snapshot / p.name).exists():
            shutil.copy2(p, snapshot / p.name)
    write(log.with_suffix('.command.json'), {'command': command, 'gpu': gpu, 'code_version': version, 'code_hashes': hashes})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
    start = time.perf_counter()
    with log.open('w') as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    write(log.with_suffix('.completion.json'), {'returncode': result.returncode, 'seconds': time.perf_counter() - start})
    if result.returncode:
        raise RuntimeError(str(log))


def checkpoint(pid, gpu):
    if pid == 'uniform_dense_bf16':
        assert (BASELINE / 'model.safetensors').exists()
        return BASELINE
    model = TEMP / 'checkpoints' / pid
    if (model / 'model.safetensors').exists():
        return model
    while shutil.disk_usage('/dev/shm').free < 22 * 1024**3:
        time.sleep(10)
    command = [sys.executable, RUN / 'code/export_model.py', '--model-path', MODEL,
        '--policy-json', RUN / f'configs/policies/{pid}.json', '--output-dir', model, '--cutlass-wrapper-path', WRAPPER]
    content = (RUN / f'configs/policies/{pid}.json').read_text()
    for method in ['sparse_bf16', 'sparse_nvfp4']:
        if method in content:
            command += ['--canonical-' + method.replace('_', '-') + '-state',
                Path('/dev/shm/daso_rebuttal_001_20261025/canonical') / method / 'model.pt']
    invoke(command, pid + '_export', gpu)
    destination = RUN / f'results/exports/{pid}'
    destination.mkdir(parents=True, exist_ok=True)
    for p in model.glob('*.json'):
        shutil.copy2(p, destination / p.name)
    return model


def cleanup(model, label):
    if not model.is_relative_to(TEMP):
        return
    path = model / 'model.safetensors'
    if path.exists():
        record = {'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                  'reason': 'Task-owned reproducible export; current evaluation finished; policies/commands/metadata retained.'}
        path.unlink()
        write(RUN / f'results/cleanup/{label}_{time.time_ns()}.json', record)


def quality_job(job, gpu):
    pid, kind = job['policy'], job['kind']
    done = RUN / f'results/complete/{pid}_{kind}.json'
    if done.exists():
        return
    model = checkpoint(pid, gpu)
    if kind == 'tasks':
        policy = RUN / f'configs/policies/{pid}.json'
        manifest = RUN / f'configs/manifest_{pid}.json'
        write(manifest, {'model_path': str(MODEL), 'policies': [{'label': pid, 'kind': 'ours',
            'checkpoint': str(model), 'policy_json': str(policy), 'policy_sha256': hashlib.sha256(policy.read_bytes()).hexdigest()}]})
        invoke([sys.executable, RUN / 'code/run_workflow.py', 'tasks', '--manifest', manifest, '--policy', pid,
            '--output-root', RUN / 'results/tasks', '--vllm-root', REPO.parent / 'test/vllm',
            '--cutlass-wrapper-path', WRAPPER, '--audit'], pid + '_full_tasks', gpu)
    else:
        for task in ['cnn_dm_1000', 'dsum', 'IWSLT']:
            target = RUN / f'results/generation/{pid}/{task}'
            if (target / 'metrics_0000_end.json').exists():
                continue
            answer = target / 'answers_0000_end.jsonl'
            if answer.exists():
                answer.rename(answer.with_name('answers_failed_' + str(time.time_ns()) + '.jsonl'))
            invoke([sys.executable, RUN / 'code/generate_tasks.py', '--policy', pid, '--checkpoint', model,
                '--dataset', task], pid + '_full_' + task, gpu)
    write(done, {'complete': True, 'gpu': gpu, 'job': job})
    cleanup(model, pid + '_quality')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=int)
    args = parser.parse_args()
    design = read(RUN / 'configs/design.json')
    if args.job is not None:
        quality_job(design['quality_jobs'][args.job], design['quality_gpus'][args.job])
        return
    state = RUN / 'results/worker.json'
    write(state, {'status': 'running', 'stage': 'quality', 'pid': os.getpid()})
    try:
        children = []
        for index in range(len(design['quality_jobs'])):
            log = (RUN / f'logs/quality_worker_{index}_{time.time_ns()}.log').open('w')
            children.append((subprocess.Popen([sys.executable, __file__, '--job', str(index)], stdout=log, stderr=subprocess.STDOUT), log))
        returns = []
        for child, log in children:
            returns.append(child.wait())
            log.close()
        assert all(r == 0 for r in returns), returns
        write(state, {'status': 'running', 'stage': 'speed', 'pid': os.getpid()})
        for spec in design['rounds']:
            for index, pid in enumerate([*spec['order'], 'uniform_dense_bf16']):
                end = index == len(spec['order'])
                name = pid + ('_bracket_end' if end else '')
                target = RUN / f'results/speed/node0/round{spec["round"]}/{name}.json'
                if target.exists():
                    continue
                # No other GPU computation during the final speed campaigns.
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
                    time.sleep(10)
                model = checkpoint(pid, 0)
                invoke([sys.executable, RUN / 'code/benchmark.py', '--checkpoint', model, '--policy', pid,
                    '--round', str(spec['round']), '--node', '0', '--record-id', name], f'round{spec["round"]}_{name}', 0)
                cleanup(model, f'round{spec["round"]}_{name}')
                print('speed complete', spec['round'], name, flush=True)
        write(state, {'status': 'complete', 'stage': 'all', 'pid': os.getpid()})
    except Exception as error:
        write(state, {'status': 'failed', 'error': repr(error), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
