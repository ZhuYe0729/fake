"""Complete independently frozen baseline sensitivity labels before quiet speed."""
import argparse
import fcntl
import subprocess
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write


def worker(index, override_gpu=None):
    from telemetry import nvml
    gpu = [0, 1, 4, 7][index] if override_gpu is None else override_gpu
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
    jobs = read(RUN / 'configs/hawq_robustness.json')['jobs'][index::4]
    for job in jobs:
        pid = job['policy']
        scenarios = [s for s in sorted(set(job['scenarios'])) if not (RUN / f'results/test/{pid}/{s}.json').exists()]
        if not scenarios:
            continue
        while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512 * 1024**2:
            time.sleep(10)
        with (RUN / 'results/export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            model = checkpoint(pid, gpu)
        invoke([sys.executable, RUN / 'code/quality.py', '--checkpoint', model,
            '--policy', pid, '--split', 'test', '--scenarios', *scenarios], pid + '_robustness_test', gpu)
        cleanup(model, pid + '_robustness_test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.gpu)
        return
    assert read(RUN / 'results/full_quality_complete.json')['complete']
    children = []
    for index in range(4):
        handle = (RUN / f'logs/hawq_robustness_worker{index}_{time.time_ns()}.log').open('w')
        p = subprocess.Popen([sys.executable, __file__, '--worker', str(index)], stdout=handle, stderr=subprocess.STDOUT)
        children.append((p, handle))
    codes = [p.wait() for p, _ in children]
    for _, handle in children:
        handle.close()
    assert all(code == 0 for code in codes), codes
    write(RUN / 'results/hawq_robustness_quality_complete.json', {'complete': True})


if __name__ == '__main__':
    main()
