"""Two quality-only workers over the frozen three-family downstream jobs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from calibration_queue import digest, invoke
from common import RUN, write_json
from policy_runtime import checkpoint


def worker(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    formal = json.loads((RUN / 'configs/formal_points.json').read_text())
    with (RUN / f'gpu{gpu}.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        for case in formal['unique_cases']:
            jobs = [job for job in formal['downstream_jobs'] if job['case_key'] == case['case_key']]
            if all((RUN / f"results/downstream_complete/{job['id']}.json").exists() for job in jobs):
                continue
            lock_path = RUN / f"results/downstream_locks/{case['case_key']}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open('a') as group_lock:
                try: fcntl.flock(group_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: continue
                if all((RUN / f"results/downstream_complete/{job['id']}.json").exists() for job in jobs):
                    continue
                policy = case['policy_id']
                label = f'downstream_{case["case_key"]}_g{gpu}_{time.time_ns()}'
                with checkpoint(policy, label) as root:
                    for job in jobs:
                        complete = RUN / f"results/downstream_complete/{job['id']}.json"
                        if complete.exists():
                            assert json.loads(complete.read_text())['complete']
                            continue
                        attempt = str(time.time_ns())
                        command = [sys.executable, RUN / 'code/quality_task.py', '--checkpoint', root,
                            '--policy', policy, '--task', job['task'], '--attempt', attempt, '--begin', job['begin']]
                        if job['end'] is not None:
                            command += ['--end', job['end']]
                        invoke(command, job['id'] + '_' + attempt)
                        result = RUN / f"results/tasks/{policy}/{job['task']}/{attempt}/result.json"
                        record = json.loads(result.read_text())
                        assert record['complete'] and record['policy'] == policy and record['task'] == job['task']
                        write_json(complete, {'complete': True, 'job': job, 'result': str(result),
                            'result_sha256': digest(result), 'gpu': gpu})
                        print('complete', job['id'], 'GPU', gpu, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    assert json.loads((RUN / 'results/formal_speed_complete.json').read_text())['complete']
    if args.worker is not None:
        return worker(args.worker)
    formal = json.loads((RUN / 'configs/formal_points.json').read_text())
    with (RUN / 'downstream_driver.lock').open('a') as driver_lock:
        fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        for gpu in [0, 1]:
            label = f'downstream_worker{gpu}_{time.time_ns()}'
            log = (RUN / f'logs/{label}.log').open('w')
            command = ['numactl', '--all', f'--physcpubind={0 if gpu == 0 else 32}-{7 if gpu == 0 else 39}',
                f'--membind={gpu}', sys.executable, __file__, '--worker', str(gpu)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
            children.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        while any(process.poll() is None for process, _ in children):
            completed = sum((RUN / f"results/downstream_complete/{job['id']}.json").exists()
                            for job in formal['downstream_jobs'])
            write_json(RUN / 'results/downstream_status.json', {'status': 'running',
                'complete_jobs': completed, 'total_jobs': len(formal['downstream_jobs']), 'unix': time.time()})
            time.sleep(10)
        assert all(process.returncode == 0 for process, _ in children)
        for _, log in children:
            log.close()
        for job in formal['downstream_jobs']:
            path = RUN / f"results/downstream_complete/{job['id']}.json"
            body = json.loads(path.read_text()); assert body['complete']
            assert digest(Path(body['result'])) == body['result_sha256']
        for case in formal['unique_cases']:
            if case['scenario'] != 'prefill_decode':
                continue
            for task in ['cnn_dm_1000', 'dsum', 'IWSLT']:
                output = RUN / f"results/task_metrics/{case['policy_id']}/{task}.json"
                if output.exists():
                    assert json.loads(output.read_text())['complete']
                    continue
                jobs = [job for job in formal['downstream_jobs'] if job['case_key'] == case['case_key'] and job['task'] == task]
                results = [json.loads((RUN / f"results/downstream_complete/{job['id']}.json").read_text())['result'] for job in jobs]
                invoke([sys.executable, RUN / 'code/score_generation.py', '--policy', case['policy_id'],
                    '--task', task, '--results', *results], f'score_{case["case_key"]}_{task}_{time.time_ns()}')
        write_json(RUN / 'results/all_downstream_complete.json', {'complete': True,
            'jobs': len(formal['downstream_jobs']), 'formal_points_sha256': digest(RUN / 'configs/formal_points.json'),
            'scope': 'Full original datasets and metrics; generation max256 with normal EOS. No standalone extra baseline rows.'})


if __name__ == '__main__':
    main()
