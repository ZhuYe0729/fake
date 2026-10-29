"""Correctness/memory gate on an idle card; no timing enters method selection."""
import fcntl
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write


def main():
    from telemetry import nvml
    gpu = 3
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
    assert nvml.nvmlDeviceGetMemoryInfo(handle).used < 512 * 1024**2
    template = read(RUN / 'configs/policies/graph_phase_dense_nvfp4_sq_int8.json')
    pid = 'smoke_phase_dense_nvfp4_sq_int8'
    write(RUN / f'configs/policies/{pid}.json', {**template, 'policy_id': pid})
    config = read(RUN / 'configs/representative_workloads.json')
    one = config['workloads'][0]
    with (RUN / 'results/export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = checkpoint(pid, gpu)
    labels = []
    for mode in ['arena_eager', 'arena_graph']:
        label = f'smoke_{one["label"]}_{mode}'
        labels.append(label)
        invoke([sys.executable, RUN / 'code/benchmark.py', '--checkpoint', model,
            '--policy', pid, '--mode', mode, '--label', label, '--outputs', str(one['output']),
            '--iterations', '1', '--batch-size', str(one['batch']), '--prompt-length', str(one['prompt']),
            '--input-file', one['input'], '--audit-weights'], label, gpu)
    first, second = [read(RUN / f'results/measurements/{label}.json') for label in labels]
    assert [x['output_token_ids'] for x in first['cases'][0]['samples']] == [x['output_token_ids'] for x in second['cases'][0]['samples']]
    assert all(x['exact_equal'] for d in [first, second] for x in d['exact_weight_audit'].values())
    cleanup(model, 'smoke_mixed_b1')
    long = config['workloads'][-1]
    baseline = checkpoint('uniform_dense_bf16', gpu)
    label = f'smoke_{long["label"]}_bf16'
    invoke([sys.executable, RUN / 'code/benchmark.py', '--checkpoint', baseline,
        '--policy', 'uniform_dense_bf16', '--mode', 'arena_graph', '--label', label,
        '--outputs', str(long['output']), '--iterations', '1', '--batch-size', str(long['batch']),
        '--prompt-length', str(long['prompt']), '--input-file', long['input']], label, gpu)
    write(RUN / 'results/workload_smoke_gate.json', {'passed': True, 'gpu': gpu,
        'mixed_b1_tokens_equal': True, 'all_moved_weights_exact': True,
        'long_b8_bf16_memory_feasible': True,
        'scope': 'Correctness/memory smoke only, one timed sample and concurrent quality elsewhere. Timings excluded from selection and rankings.'})


if __name__ == '__main__':
    main()
