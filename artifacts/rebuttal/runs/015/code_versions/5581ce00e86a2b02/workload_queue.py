"""Quiet transfer timing after the main campaigns finish."""
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write


def main():
    assert read(RUN / 'results/formal_progress.json')['status'] == 'complete'
    config = read(RUN / 'configs/representative_workloads.json')
    from telemetry import nvml
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    for workload in config['workloads']:
        for number, order in enumerate(config['rounds']):
            other = [p for p in order if p != 'uniform_dense_bf16']
            for index, pid in enumerate(['uniform_dense_bf16', *other, 'uniform_dense_bf16']):
                suffix = '_bracket_end' if index == len(other) + 1 else ''
                label = f'workload_{workload["label"]}_round{number}_{pid}{suffix}'
                if (RUN / f'results/measurements/{label}.json').exists():
                    continue
                while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512 * 1024**2:
                    time.sleep(10)
                model = checkpoint(pid, 0)
                write(RUN / 'results/workload_progress.json', {'status': 'running',
                    'workload': workload['label'], 'round': number, 'policy': pid})
                invoke([sys.executable, RUN / 'code/benchmark.py', '--checkpoint', model,
                    '--policy', pid, '--mode', 'arena_graph', '--label', label,
                    '--outputs', str(workload['output']), '--iterations', str(config['iterations']),
                    '--batch-size', str(workload['batch']), '--prompt-length', str(workload['prompt']),
                    '--input-file', workload['input']], label, 0)
                cleanup(model, label)
    write(RUN / 'results/workload_progress.json', {'status': 'complete'})


if __name__ == '__main__':
    main()
