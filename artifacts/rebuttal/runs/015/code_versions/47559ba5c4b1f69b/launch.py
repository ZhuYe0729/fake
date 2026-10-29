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
TEMP = Path('/dev/shm/daso_rebuttal_015_20261026')
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
    files = sorted(p for p in (RUN / 'code').rglob('*') if p.is_file())
    hashes = {str(p.relative_to(RUN / 'code')): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = RUN / 'code_versions' / version
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in files:
        target = snapshot / p.relative_to(RUN / 'code')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(p, target)
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
    if model.exists():
        previous = model.with_name(pid + '_previous_' + str(time.time_ns()))
        model.rename(previous)
        write(RUN / f'results/export_history/{pid}_{time.time_ns()}.json',
              {'previous_directory': str(previous), 'reason': 'Preserve metadata before rebuilding a cleaned export.'})
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
    attempt = str(time.time_ns())
    records = {}
    for p in model.glob('*.json'):
        target = destination / p.name
        if target.exists() and target.read_bytes() != p.read_bytes():
            target = destination / attempt / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(p, target)
        records[p.name] = {'source': str(p), 'saved': str(target), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    write(destination / f'export_attempt_{attempt}.json', records)
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



def main():
    policies = sys.argv[1:] or ['hawq_fp4_phase_pair_signed_prefill_decode_q0.07']
    for pid in policies:
        model=checkpoint(pid,0)
        for mode in ['legacy','arena_eager','arena_graph']:
            label='probe_'+pid+'_'+mode
            if (RUN/f'results/measurements/{label}.json').exists():continue
            command=[sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',pid,
                '--mode',mode,'--label',label,'--outputs','64','--iterations','3']
            if mode!='legacy':command+=['--audit-weights']
            invoke(command,label,0)
            print(label,'done',flush=True)


if __name__=='__main__':main()
