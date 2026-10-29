"""Parallel quality only; exports serialized and all formal timing deferred."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write

GPUS = [0, 3, 4, 7]


def worker(index, additional=False):
    from telemetry import nvml
    filename = 'validation_additional_jobs.json' if additional else 'validation_jobs.json'
    jobs = read(RUN / 'configs' / filename)['jobs']
    gpu = GPUS[index]
    for job in jobs[index::len(GPUS)]:
        pid = job['policy']
        scenarios = [s for s in job['scenarios']
                     if not (RUN / f'results/validation/{pid}/{s}.json').exists()]
        if not scenarios:
            continue
        handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
        while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512 * 1024**2:
            time.sleep(10)
        with (RUN / 'results/export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while shutil.disk_usage('/dev/shm').free < 26 * 1024**3:
                time.sleep(10)
            model = checkpoint(pid, gpu)
        for scenario in scenarios:
            partial = RUN / f'results/captures/validation/{pid}/{scenario}'
            if partial.exists():
                partial.rename(partial.with_name(scenario + '_failed_' + str(time.time_ns())))
        invoke([sys.executable, RUN / 'code/quality.py', '--checkpoint', model, '--policy', pid,
                '--split', 'validation', '--scenarios', *scenarios], pid + '_validation', gpu)
        write(RUN / f'results/validation_complete/{pid}.json', {'complete': True, 'gpu': gpu, 'job': job})
        cleanup(model, pid + '_validation')
        print('validation complete', pid, scenarios, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--additional', action='store_true')
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.additional)
        return
    status = RUN / 'results/validation_queue.json'
    children = []
    started = time.perf_counter()
    for index, gpu in enumerate(GPUS):
        handle = (RUN / f'logs/validation_worker{index}_{time.time_ns()}.log').open('w')
        command = [sys.executable, __file__, '--worker', str(index)] + (['--additional'] if args.additional else [])
        process = subprocess.Popen(command,
                                   stdout=handle, stderr=subprocess.STDOUT)
        children.append((process, handle, gpu))
    while any(p.poll() is None for p, _, _ in children):
        write(status, {'status': 'running', 'workers': [{'pid': p.pid, 'gpu': gpu, 'returncode': p.poll()}
                       for p, _, gpu in children], 'seconds': time.perf_counter() - started})
        time.sleep(10)
    returns = [p.returncode for p, _, _ in children]
    for _, handle, _ in children:
        handle.close()
    write(status, {'status': 'complete' if all(x == 0 for x in returns) else 'failed',
                   'returncodes': returns, 'seconds': time.perf_counter() - started})
    assert all(x == 0 for x in returns), returns


if __name__ == '__main__':
    main()
