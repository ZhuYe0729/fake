"""Quiet actual E2E calibration for the prespecified DASO validation pool."""
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write


def main():
    assert read(RUN / 'results/validation_complete.json')['complete']
    pool = read(RUN / 'configs/allocation_pool.json')
    from telemetry import nvml
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    pids = ['uniform_dense_bf16', *sorted(p for p, x in pool['policies'].items()
        if x['selection_scenarios'] and p != 'uniform_dense_bf16')]
    for index, pid in enumerate(pids):
        label = 'calibration_' + pid
        if (RUN / f'results/measurements/{label}.json').exists():
            continue
        while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512 * 1024**2:
            time.sleep(10)
        model = checkpoint(pid, 0)
        scenarios = pool['policies'][pid]['selection_scenarios']
        outputs = [o for s, o in [('prefill_only', 1), ('prefill_decode', 64)] if s in scenarios]
        write(RUN / 'results/calibration_progress.json', {'status': 'running', 'index': index, 'total': len(pids), 'policy': pid})
        invoke([sys.executable, RUN / 'code/benchmark.py', '--checkpoint', model, '--policy', pid,
            '--mode', 'arena_graph', '--label', label, '--outputs', *map(str, outputs),
            '--iterations', '5', '--input-file', RUN / 'configs/calibration_speed_inputs.json'], label, 0)
        cleanup(model, label)
    write(RUN / 'results/calibration_complete.json', {'complete': True, 'policies': len(pids)})


if __name__ == '__main__':
    main()
