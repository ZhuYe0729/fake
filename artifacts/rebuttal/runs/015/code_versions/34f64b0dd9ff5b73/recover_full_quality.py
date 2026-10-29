"""Resume three NUMA launch failures while preserving the valid worker0."""
import subprocess
import sys
import time

from launch import RUN, invoke, read, write


def main():
    children = []
    started = time.perf_counter()
    for index in [1, 2, 3]:
        handle = (RUN / f'logs/full_quality_recovery_worker{index}_{time.time_ns()}.log').open('w')
        process = subprocess.Popen([sys.executable, RUN / 'code/full_quality_queue.py',
            '--worker', str(index)], stdout=handle, stderr=subprocess.STDOUT)
        children.append((process, handle))
    jobs = read(RUN / 'configs/full_quality_jobs.json')['jobs']
    while True:
        done = sum((RUN / f'results/full_quality_complete/{j["policy"]}.json').exists() for j in jobs)
        write(RUN / 'results/full_quality_recovery.json', {'status': 'running', 'complete': done,
            'total': len(jobs), 'seconds': time.perf_counter() - started,
            'workers': [{'pid': p.pid, 'returncode': p.poll()} for p, _ in children],
            'reason': 'Nested numactl inherited CPU0-7 and rejected other NUMA CPUs. Added --all. Worker0 retained; workers1/2/3 resumed on idle GPU1/4/7.'})
        if all(p.poll() is not None for p, _ in children):
            assert all(p.returncode == 0 for p, _ in children), [p.returncode for p, _ in children]
            if done == len(jobs):
                break
        time.sleep(10)
    for _, handle in children:
        handle.close()
    write(RUN / 'results/full_quality_complete.json', {'complete': True,
        'recovered': True, 'seconds': time.perf_counter() - started})
    # The original parent records its launch failures before exiting.
    while read(RUN / 'results/full_quality_queue.json')['status'] == 'running':
        time.sleep(10)
    write(RUN / 'results/full_quality_recovery.json', {'status': 'complete', 'jobs': len(jobs)})
    invoke([sys.executable, RUN / 'code/post_selection.py'], 'post_selection_recovered', 0)


if __name__ == '__main__':
    main()
