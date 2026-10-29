"""Run target-model feature/Hessian preparation after Qwen formal speed ends."""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
REBUTTAL = RUN.parents[1]


def main():
    while not (RUN / 'results/cpu_preparation_complete.json').exists():
        time.sleep(10)
    from launch import invoke, write, read
    from telemetry import nvml
    previous = REBUTTAL / 'runs/016'
    while not (previous / 'results/formal_complete.json').exists():
        write(RUN / 'results/gpu_preparation_progress.json', {'status': 'waiting', 'reason': 'run016 formal speed'})
        time.sleep(10)
    locks = REBUTTAL / 'night151/gpu_locks'
    locks.mkdir(parents=True, exist_ok=True)
    claim = None
    while claim is None:
        if not (previous / 'results/full_quality_progress.json').exists() and not (previous / 'results/full_quality_complete.json').exists():
            time.sleep(10)
            continue
        reserved = set()
        if not (previous / 'results/full_quality_complete.json').exists():
            reserved.update(read(previous / 'results/full_quality_progress.json').get('gpus', []))
        for gpu in range(nvml.nvmlDeviceGetCount()):
            if gpu in reserved or nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used >= 512*1024**2:
                continue
            candidate = (locks / f'gpu{gpu}.lock').open('a')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            claim = candidate
            break
        if claim is None:
            time.sleep(10)
    try:
        write(RUN / 'results/gpu_preparation_progress.json', {'status': 'running', 'gpu': gpu, 'pid': os.getpid()})
        if not (RUN / 'results/features_complete.json').exists():
            invoke([sys.executable, RUN / 'code/collect_features.py'], 'target_model_features', gpu)
        hessian = RUN / 'results/hessian_trace.json'
        if not hessian.exists() or not read(hessian)['complete']:
            invoke([sys.executable, RUN / 'code/hessian_trace.py', '--probes', '16', '--blocks', '4',
                '--sequence-length', '256', '--layers-per-group', '2', '--output', hessian], 'target_model_hessian', gpu)
        write(RUN / 'results/gpu_preparation_complete.json', {'complete': True, 'gpu': gpu})
    finally:
        claim.close()


if __name__ == '__main__':
    main()
