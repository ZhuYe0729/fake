"""Wait for the already-running controls, then finish stronger paired checks."""
import json
import sys
import time

from launch import RUN, invoke


def main():
    expected = [f'round{r}_{p}_{mode}' for r in range(2)
                for p in ['uniform_dense_bf16', 'uniform_w4a16_ours', 'uniform_marlin_int4']
                for mode in ['eager', 'eager_frozen', 'graph']]
    while True:
        completed = []
        for label in expected:
            for path in (RUN / 'logs').glob(label + '_*.completion.json'):
                # Prefixes such as eager_frozen must not match the eager label.
                command = json.loads(path.with_name(path.name.replace('.completion.json', '.command.json')).read_text())
                argv = command['command']
                if '--label' in argv and argv[argv.index('--label') + 1] == label:
                    assert json.loads(path.read_text())['returncode'] == 0, str(path)
                    completed.append(label)
        if set(completed) == set(expected):
            break
        time.sleep(10)
    for policy in ['uniform_w4a16_ours', 'uniform_dense_bf16', 'uniform_marlin_int4']:
        if not (RUN / f'results/paired/{policy}.json').exists():
            invoke([sys.executable, RUN / 'code/paired_generation.py', policy], 'paired_' + policy)
    invoke([sys.executable, RUN / 'code/analyze.py'], 'analysis')


if __name__ == '__main__':
    main()
