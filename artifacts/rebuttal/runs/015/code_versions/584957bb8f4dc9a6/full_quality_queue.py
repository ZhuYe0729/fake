"""Complete changed-map downstream tasks and held-out NLL, then release GPUs."""
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time

from launch import RUN, REPO, MODEL, WRAPPER, checkpoint, cleanup, invoke, read, write

GPUS = [0, 3, 4, 7]


def worker(index):
    from telemetry import nvml
    gpu = GPUS[index]
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
    for job in read(RUN / 'configs/full_quality_jobs.json')['groups'][index]:
        pid = job['policy']
        if (RUN / f'results/full_quality_complete/{pid}.json').exists():
            continue
        while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512 * 1024**2:
            time.sleep(10)
        with (RUN / 'results/export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while shutil.disk_usage('/dev/shm').free < 26 * 1024**3:
                time.sleep(10)
            model = checkpoint(pid, gpu)
        for kind in job['kinds']:
            marker = RUN / f'results/complete/{pid}_{kind}.json'
            if marker.exists():
                continue
            if kind == 'tasks':
                manifest = RUN / f'configs/manifest_{pid}.json'
                write(manifest, {'model_path': str(MODEL), 'policies': [{'label': pid, 'kind': 'ours',
                    'checkpoint': str(model), 'policy_json': str(RUN / f'configs/policies/{pid}.json')}]})
                invoke([sys.executable, RUN / 'code/run_workflow.py', 'tasks', '--manifest', manifest,
                    '--policy', pid, '--output-root', RUN / 'results/tasks', '--vllm-root', REPO.parent / 'test/vllm',
                    '--cutlass-wrapper-path', WRAPPER, '--audit'], pid + '_full_tasks', gpu)
            else:
                for dataset in ['cnn_dm_1000', 'dsum', 'IWSLT']:
                    target = RUN / f'results/generation/{pid}/{dataset}'
                    if (target / 'metrics_0000_end.json').exists():
                        continue
                    partial = target / 'answers_0000_end.jsonl'
                    if partial.exists():
                        partial.rename(partial.with_name('answers_failed_' + str(time.time_ns()) + '.jsonl'))
                    invoke([sys.executable, RUN / 'code/generate_tasks.py', '--checkpoint', model,
                            '--policy', pid, '--dataset', dataset], pid + '_full_' + dataset, gpu)
            write(marker, {'complete': True, 'policy': pid, 'kind': kind})
        scenarios = [s for s in job['test_scenarios'] if not (RUN / f'results/test/{pid}/{s}.json').exists()]
        if scenarios:
            for scenario in scenarios:
                partial = RUN / f'results/captures/test/{pid}/{scenario}'
                if partial.exists():
                    partial.rename(partial.with_name(scenario + '_failed_' + str(time.time_ns())))
            invoke([sys.executable, RUN / 'code/quality.py', '--checkpoint', model,
                    '--policy', pid, '--split', 'test', '--scenarios', *scenarios], pid + '_test_nll', gpu)
        write(RUN / f'results/full_quality_complete/{pid}.json', {'complete': True, 'gpu': gpu, 'job': job})
        cleanup(model, pid + '_full_quality')
        print('full quality complete', pid, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
        return
    children = []
    started = time.perf_counter()
    for index in range(4):
        handle = (RUN / f'logs/full_quality_worker{index}_{time.time_ns()}.log').open('w')
        p = subprocess.Popen([sys.executable, __file__, '--worker', str(index)], stdout=handle, stderr=subprocess.STDOUT)
        children.append((p, handle))
    while any(p.poll() is None for p, _ in children):
        write(RUN / 'results/full_quality_queue.json', {'status': 'running',
            'workers': [{'pid': p.pid, 'returncode': p.poll()} for p, _ in children],
            'seconds': time.perf_counter() - started})
        time.sleep(10)
    returns = [p.returncode for p, _ in children]
    for _, h in children:
        h.close()
    write(RUN / 'results/full_quality_queue.json', {'status': 'complete' if all(x == 0 for x in returns) else 'failed',
          'returncodes': returns, 'seconds': time.perf_counter() - started})
    assert all(x == 0 for x in returns), returns
    write(RUN / 'results/full_quality_complete.json', {'complete': True, 'seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
