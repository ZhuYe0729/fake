"""One model per idle card, with nonblocking job locks and dynamic work sharing."""
import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import time

from launch import MODEL, REPO, RUN, WRAPPER, checkpoint, cleanup, invoke, read, write


def worker(stage, gpu):
    from telemetry import nvml
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
    jobs = read(RUN / f'configs/{stage}_jobs.json')
    for job in jobs:
        label = job['job_id']
        marker = RUN / f'results/completed/{stage}/{label}.json'
        if marker.exists():
            continue
        lockfile = RUN / f'results/locks/{stage}/{label}.lock'
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        with lockfile.open('a') as claim:
            try:
                fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if marker.exists():
                continue
            while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512 * 1024**2:
                time.sleep(10)
            pid = job['policy']
            if 'policy_sha256' in job:
                assert hashlib.sha256((RUN / f'configs/policies/{pid}.json').read_bytes()).hexdigest() == job['policy_sha256']
            with (RUN / 'results/export.lock').open('a') as export:
                fcntl.flock(export, fcntl.LOCK_EX)
                while shutil.disk_usage('/dev/shm').free < 26 * 1024**3:
                    time.sleep(10)
                model = checkpoint(pid, gpu)
            if stage == 'full_quality':
                for kind in job['kinds']:
                    if kind == 'tasks':
                        manifest = RUN / f'configs/manifest_{pid}.json'
                        policy = RUN / f'configs/policies/{pid}.json'
                        write(manifest, {'model_path': str(MODEL), 'policies': [{'label': pid, 'kind': 'ours',
                            'checkpoint': str(model), 'policy_json': str(policy),
                            'policy_sha256': hashlib.sha256(policy.read_bytes()).hexdigest()}]})
                        invoke([sys.executable, RUN / 'code/run_workflow.py', 'tasks', '--manifest', manifest,
                            '--policy', pid, '--output-root', RUN / 'results/tasks', '--vllm-root', REPO.parent / 'test/vllm',
                            '--cutlass-wrapper-path', WRAPPER, '--audit'], label, gpu)
                    else:
                        for dataset in ['cnn_dm_1000', 'dsum', 'IWSLT']:
                            target = RUN / f'results/generation/{pid}/{dataset}'
                            if (target / 'metrics_0000_end.json').exists():
                                continue
                            partial = target / 'answers_0000_end.jsonl'
                            if partial.exists():
                                partial.rename(partial.with_name('answers_failed_' + str(time.time_ns()) + '.jsonl'))
                            invoke([sys.executable, RUN / 'code/generate_tasks.py', '--checkpoint', model,
                                '--policy', pid, '--dataset', dataset], label + '_' + dataset, gpu)
            else:
                split = 'training' if stage == 'training' else stage
                scenarios = []
                for scenario in job['scenarios']:
                    result = (RUN / f'results/training/{scenario}/{pid}.json' if stage == 'training' else
                              RUN / f'results/{split}/{pid}/{scenario}.json')
                    if result.exists():
                        continue
                    partial = RUN / f'results/captures/{split}/{pid}/{scenario}'
                    if partial.exists():
                        partial.rename(partial.with_name(scenario + '_failed_' + str(time.time_ns())))
                    scenarios.append(scenario)
                if scenarios:
                    invoke([sys.executable, RUN / 'code/score_nll.py', '--checkpoint', model,
                        '--policy', pid, '--split', split, '--scenarios', *scenarios], label, gpu)
            write(marker, {'complete': True, 'job': job, 'gpu': gpu})
            cleanup(model, label)
            print('quality job complete', stage, label, gpu, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.gpu is not None:
        worker(args.stage, args.gpu)
        return
    jobs = read(RUN / f'configs/{args.stage}_jobs.json')
    if all((RUN / f'results/completed/{args.stage}/{j["job_id"]}.json').exists() for j in jobs):
        write(RUN / f'results/{args.stage}_complete.json', {
            'complete': True, 'jobs': len(jobs), 'reused_completed_jobs': True})
        write(RUN / f'results/{args.stage}_progress.json', {
            'status': 'complete', 'complete': len(jobs), 'total': len(jobs)})
        return
    from telemetry import nvml
    while True:
        available = [g for g in range(nvml.nvmlDeviceGetCount())
                     if nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(g)).used < 512 * 1024**2]
        if len(available) >= 4:
            break
        time.sleep(10)
    children = []
    for gpu in available[:4]:
        handle = (RUN / f'logs/{args.stage}_worker_gpu{gpu}_{time.time_ns()}.log').open('w')
        p = subprocess.Popen([sys.executable, __file__, '--stage', args.stage, '--gpu', str(gpu)], stdout=handle, stderr=subprocess.STDOUT)
        children.append((p, handle))
    while any(p.poll() is None for p, _ in children):
        count = len(list((RUN / f'results/completed/{args.stage}').glob('*.json')))
        write(RUN / f'results/{args.stage}_progress.json', {'status': 'running', 'complete': count,
            'total': len(read(RUN / f'configs/{args.stage}_jobs.json')), 'gpus': available[:4],
            'workers': [{'pid': p.pid, 'returncode': p.poll()} for p, _ in children]})
        time.sleep(10)
    codes = [p.returncode for p, _ in children]
    for _, handle in children:
        handle.close()
    pending = [j['job_id'] for j in jobs
               if not (RUN / f'results/completed/{args.stage}/{j["job_id"]}.json').exists()]
    assert not pending, {'worker_exit_codes': codes, 'pending': pending}
    # Another worker can finish a job after its first worker failed. Keep the
    # failure evidence, but judge stage completion by completed jobs.
    write(RUN / f'results/{args.stage}_complete.json', {'complete': True,
        'jobs': len(jobs), 'gpus': available[:4], 'worker_exit_codes': codes})
    write(RUN / f'results/{args.stage}_progress.json', {
        'status': 'complete', 'complete': len(jobs), 'total': len(jobs), 'worker_exit_codes': codes})


if __name__ == '__main__':
    main()
