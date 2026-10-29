"""Finish added anchors as each original worker group completes, then time quietly."""
import subprocess
import sys
import time

from launch import RUN, invoke, read, write


def main():
    initial = read(RUN / 'configs/validation_jobs.json')['jobs']
    children = {}
    started = time.perf_counter()
    while True:
        for index in range(4):
            group = initial[index::4]
            done = all((RUN / f'results/validation/{j["policy"]}/{s}.json').exists()
                       for j in group for s in j['scenarios'])
            if done and index not in children:
                handle = (RUN / f'logs/validation_additional_worker{index}_{time.time_ns()}.log').open('w')
                process = subprocess.Popen([sys.executable, RUN / 'code/validation_queue.py',
                    '--worker', str(index), '--additional'], stdout=handle, stderr=subprocess.STDOUT)
                children[index] = (process, handle)
        write(RUN / 'results/validation_recovery.json', {'status': 'running',
            'additional_workers': {i: {'pid': p.pid, 'returncode': p.poll()} for i, (p, _) in children.items()},
            'seconds': time.perf_counter() - started,
            'resource_recovery': 'Initial GPU1/2/5 starts failed while other tasks occupied those cards; resumed on GPU0/3/7, with original GPU4 worker retained.'})
        if len(children) == 4 and all(p.poll() is not None for p, _ in children.values()):
            break
        time.sleep(10)
    returns = [p.returncode for p, _ in children.values()]
    for _, handle in children.values():
        handle.close()
    assert all(x == 0 for x in returns), returns
    pool = read(RUN / 'configs/reanalysis_pool.json')
    assert all((RUN / f'results/validation/{pid}/{s}.json').exists()
               for pid, p in pool['policies'].items() if p['daso_eligible'] for s in p['scenarios'])
    write(RUN / 'results/validation_recovery.json', {'status': 'complete', 'seconds': time.perf_counter() - started,
          'scope': 'All original and amended candidates complete; initial resource failures retained.'})
    # Every quality child has returned; calibration is now serialized on GPU0.
    invoke([sys.executable, RUN / 'code/calibration_queue.py'], 'calibration_queue', 0)
    invoke([sys.executable, RUN / 'code/select_reanalysis.py'], 'selection', 0)


if __name__ == '__main__':
    main()
