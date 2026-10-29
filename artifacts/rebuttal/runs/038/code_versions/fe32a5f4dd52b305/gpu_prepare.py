"""Run target-model feature/Hessian preparation after Qwen formal speed ends."""
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
    previous = REBUTTAL / 'runs/016'
    while not (previous / 'results/formal_complete.json').exists():
        write(RUN / 'results/gpu_preparation_progress.json', {'status': 'waiting', 'reason': 'run016 formal speed'})
        time.sleep(10)
    from resources import quality_gpus
    with quality_gpus(maximum=1, minimum=1) as cards:
        gpu = cards[0]
        write(RUN / 'results/gpu_preparation_progress.json', {'status': 'running', 'gpu': gpu, 'pid': os.getpid()})
        if not (RUN / 'results/features_complete.json').exists():
            invoke([sys.executable, RUN / 'code/collect_features.py'], 'target_model_features', gpu)
        hessian = RUN / 'results/hessian_trace.json'
        if not hessian.exists() or not read(hessian)['complete']:
            invoke([sys.executable, RUN / 'code/hessian_trace.py', '--probes', '16', '--blocks', '4',
                '--sequence-length', '256', '--layers-per-group', '2', '--output', hessian], 'target_model_hessian', gpu)
        write(RUN / 'results/gpu_preparation_complete.json', {'complete': True, 'gpu': gpu})


if __name__ == '__main__':
    main()
