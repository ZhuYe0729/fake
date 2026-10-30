"""Two GPU workers for independent policy-quality closures, never timing."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import RUN, write_json
from policy_runtime import digest, quality


def worker(gpu, manifest):
    body = json.loads(manifest.read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    with (RUN / f'gpu{gpu}.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        for index, job in enumerate(body['jobs']):
            target = RUN / f"results/quality_pool/{body['stage']}/{index:03d}.json"
            lock_path = target.with_suffix('.lock')
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open('a') as job_lock:
                try:
                    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if target.exists():
                    assert json.loads(target.read_text())['manifest_sha256'] == digest(manifest)
                    continue
                label = f"{body['stage']}_{job['policy_id']}_g{gpu}_{time.time_ns()}"
                paths = quality(job['policy_id'], body['split'], label, job['scenario'])
                write_json(target, {'complete': True, 'job': job, 'gpu': gpu,
                    'manifest_sha256': digest(manifest), 'results': [
                        {'path': str(path), 'sha256': digest(path)} for path in paths]})
                print('complete', body['stage'], job['policy_id'], job['scenario'], 'GPU', gpu, flush=True)


def run(jobs, split, stage):
    jobs = list({(row['policy_id'], row['scenario']): row for row in jobs}.values())
    manifest = RUN / f'configs/{stage}_quality_jobs.json'
    body = {'stage': stage, 'split': split, 'jobs': jobs}
    if manifest.exists():
        assert json.loads(manifest.read_text()) == body
    else:
        write_json(manifest, body)
    children = []
    for gpu in [0, 1]:
        log = (RUN / f'logs/{stage}_worker{gpu}_{time.time_ns()}.log').open('w')
        command = ['numactl', '--all', f'--physcpubind={0 if gpu == 0 else 32}-{7 if gpu == 0 else 39}',
            f'--membind={gpu}', sys.executable, __file__, '--worker', str(gpu), '--manifest', str(manifest)]
        children.append((subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
            stdout=log, stderr=subprocess.STDOUT), log))
    while any(process.poll() is None for process, _ in children):
        write_json(RUN / f'results/{stage}_pool_status.json', {'stage': stage, 'jobs': len(jobs),
            'complete_jobs': sum((RUN / f'results/quality_pool/{stage}/{index:03d}.json').exists() for index in range(len(jobs))),
            'unix': time.time()})
        time.sleep(10)
    assert all(process.returncode == 0 for process, _ in children), f'{stage}: inspect worker logs'
    for _, log in children:
        log.close()
    paths = []
    for index, job in enumerate(jobs):
        closed = json.loads((RUN / f'results/quality_pool/{stage}/{index:03d}.json').read_text())
        assert closed['complete'] and closed['manifest_sha256'] == digest(manifest) and closed['job'] == job
        for record in closed['results']:
            path = Path(record['path'])
            assert digest(path) == record['sha256']
            result = json.loads(path.read_text())
            assert result['complete'] and len(result['blocks']) == 100
            paths.append(str(path))
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=[0, 1], required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    worker(args.worker, args.manifest)
