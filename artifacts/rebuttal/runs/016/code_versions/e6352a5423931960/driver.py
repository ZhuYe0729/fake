"""Finish run015 quietly, then execute the frozen matched-candidate study."""
import hashlib
import statistics
import sys
import time

from launch import RUN, checkpoint, invoke, read, write


def stage(name, script, *arguments):
    write(RUN / 'results/worker.json', {'status': 'running', 'stage': name})
    invoke([sys.executable, RUN / 'code' / script, *arguments], name, 0)


def idle_gpu():
    from telemetry import nvml
    while True:
        for gpu in range(nvml.nvmlDeviceGetCount()):
            if nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used < 512 * 1024**2:
                return gpu
        time.sleep(10)


def main():
    previous = RUN.parent / '015/results/final_audit.json'
    while not previous.exists():
        write(RUN / 'results/worker.json', {'status': 'waiting', 'stage': 'run015_final_audit',
            'reason': 'No new GPU jobs overlap run015 formal or representative speed campaigns.'})
        time.sleep(10)
    assert read(previous)['verified']
    if not (RUN / 'reference/protected_run015_manifest.json').exists():
        write(RUN / 'reference/protected_run015_manifest.json', {
            'sha256': hashlib.sha256((RUN.parent / '015/results/manifest.json').read_bytes()).hexdigest(),
            'scope': 'Prior archived evidence retained; run016 writes only its own tree and a new response folder.'})
    if not (RUN / 'results/int8_local_errors_complete.json').exists():
        gpu = idle_gpu()
        write(RUN / 'results/worker.json', {'status': 'running', 'stage': 'plain_int8_local_errors', 'gpu': gpu})
        invoke([sys.executable, RUN / 'code/collect_int8_errors.py'], 'plain_int8_local_errors', gpu)
    if not (RUN / 'results/training_reference_compatible.json').exists():
        checks = []
        for scenario in ['prefill_only', 'prefill_decode']:
            target = RUN / f'results/training/{scenario}/uniform_dense_bf16.json'
            if not target.exists():
                gpu = idle_gpu()
                model = checkpoint('uniform_dense_bf16', gpu)
                invoke([sys.executable, RUN / 'code/score_nll.py', '--checkpoint', model,
                    '--policy', 'uniform_dense_bf16', '--split', 'training', '--scenarios', scenario],
                    'training_bf16_' + scenario, gpu)
            manifest = read(RUN / f'configs/calibration_{scenario}.json')
            old = read(RUN / f'results/training/{scenario}/{manifest[0]["policy_id"]}.json')
            new = read(target)
            differences = [a['avg_nll']-b['avg_nll'] for a,b in zip(new['blocks'],old['blocks'],strict=True)]
            assert len(differences) == 100
            checks.append({'scenario': scenario, 'mean_difference': statistics.mean(differences),
                'max_absolute_block_difference': max(map(abs, differences))})
            assert abs(checks[-1]['mean_difference']) < 1e-4 and checks[-1]['max_absolute_block_difference'] < 1e-3, checks[-1]
        write(RUN / 'results/training_reference_compatible.json', {'passed': True, 'checks': checks})
    if not (RUN / 'results/training_complete.json').exists():
        stage('training_labels', 'job_queue.py', '--stage', 'training')
    if not (RUN / 'results/quality_model_prefill_decode.json').exists():
        stage('quality_fit', 'fit_quality.py')
    if not (RUN / 'configs/allocation_pool.json').exists():
        stage('matched_allocations', 'allocate_suite.py')
    if not (RUN / 'results/validation_complete.json').exists():
        stage('validation_quality', 'job_queue.py', '--stage', 'validation')
    if not (RUN / 'results/calibration_complete.json').exists():
        stage('speed_calibration', 'calibration_queue.py')
    if not (RUN / 'configs/selected_policies.json').exists():
        stage('freeze_selection', 'select_suite.py')
    if not (RUN / 'configs/full_quality_jobs.json').exists():
        stage('full_task_design', 'prepare_full_quality.py')
    if not (RUN / 'results/confirmation_complete.json').exists():
        stage('confirmation_quality', 'job_queue.py', '--stage', 'confirmation')
    if not (RUN / 'results/formal_complete.json').exists():
        stage('formal_speed', 'formal_queue.py')
    stage('matched_nll_speed_report', 'build_report.py')
    if not (RUN / 'results/full_quality_complete.json').exists():
        stage('full_task_quality', 'job_queue.py', '--stage', 'full_quality')
    stage('complete_report', 'build_report.py', '--full')
    write(RUN / 'results/worker.json', {'status': 'complete', 'stage': 'all_experiments_and_reports'})
    stage('audit', 'audit.py')
    write(RUN / 'results/worker.json', {'status': 'complete', 'stage': 'audited'})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state = read(RUN / 'results/worker.json')
        write(RUN / 'results/worker.json', {**state, 'status': 'failed',
            'error': repr(error), 'failed_at_unix': time.time()})
        raise
