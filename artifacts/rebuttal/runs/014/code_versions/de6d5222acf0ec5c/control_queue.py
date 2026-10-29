"""Predeclared interleaved uniform-policy generation controls."""
import sys
from pathlib import Path

from launch import RUN, TEMP, invoke

POLICIES = ['uniform_dense_bf16', 'uniform_w4a16_ours', 'uniform_marlin_int4']


def main():
    selection = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if selection == 'probe':
        jobs = [(0, 'uniform_w4a16_ours', 'graph')]
    else:
        jobs = [(r, p, mode) for r in range(2)
                for p in (POLICIES if r == 0 else POLICIES[::-1])
                for mode in (['eager', 'eager_frozen', 'graph'] if r == 0 else ['graph', 'eager_frozen', 'eager'])]
    for r, policy, mode in jobs:
        label = f'round{r}_{policy}_{mode}' + ('_probe' if selection == 'probe' else '')
        if (RUN / f'results/generation_controls/{label}.json').exists():
            continue
        checkpoint = (Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_dense_bf16')
                      if policy == 'uniform_dense_bf16' else TEMP / 'checkpoints' / policy)
        command = [sys.executable, RUN / 'code/generation_control.py', '--checkpoint', checkpoint,
                   '--policy', policy, '--mode', mode, '--label', label]
        if r == 0 and mode == 'eager' and selection != 'probe':
            command.append('--profile')
        invoke(command, label)


if __name__ == '__main__':
    main()
