"""Three frozen same-card campaigns after all quality evaluation ends."""
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write


def main():
    assert read(RUN / 'results/full_quality_complete.json')['complete']
    assert read(RUN / 'results/hawq_robustness_quality_complete.json')['complete']
    design = read(RUN / 'configs/design.json')
    from telemetry import nvml
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    for spec in design['rounds']:
        order = [*spec['order'], 'uniform_dense_bf16']
        for index, pid in enumerate(order):
            end = index == len(order) - 1
            name = pid + ('_bracket_end' if end else '')
            target = RUN / f'results/speed/node0/round{spec["round"]}/{name}.json'
            if target.exists():
                continue
            while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512 * 1024**2:
                time.sleep(10)
            model = checkpoint(pid, 0)
            write(RUN / 'results/formal_progress.json', {'status': 'running', 'round': spec['round'],
                  'index': index, 'total_policies': len(order), 'policy': pid})
            invoke([sys.executable, RUN / 'code/formal_benchmark.py', '--checkpoint', model,
                    '--policy', pid, '--round', str(spec['round']), '--record-id', name,
                    '--outputs', *map(str, design['policies'][pid]['outputs'])],
                   f'formal_round{spec["round"]}_{name}', 0)
            cleanup(model, f'formal_round{spec["round"]}_{name}')
            print('formal complete', spec['round'], name, flush=True)
    write(RUN / 'results/formal_progress.json', {'status': 'complete'})


if __name__ == '__main__':
    main()
