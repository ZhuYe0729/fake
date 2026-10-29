"""Complete the original-model core rebuttal table with supervised restart markers."""
import hashlib
import shutil
import sys
import time

from launch import RUN, checkpoint, invoke, read, write


def state(stage, status='running', **details):
    write(RUN / 'results/worker.json', {'status': status, 'stage': stage, **details})


def wait_for(path, label):
    while not path.exists():
        state(label, 'waiting')
        time.sleep(10)


def stage(label, script, *args):
    cpu_only = script in ['fit_quality.py','allocate_suite.py','select_suite.py',
                          'freeze_primary.py','report.py','audit_core.py']
    state(label, execution='cpu_only' if cpu_only else 'gpu_or_queue')
    invoke([sys.executable, RUN / 'code' / script, *args], label, 6 if cpu_only else 0)


def main():
    wait_for(RUN / 'results/cpu_preparation_complete.json', 'cpu_preparation')
    assert read(RUN / 'configs/frozen_design.json')['generation_protocol_sha256'] == hashlib.sha256(
        (RUN / 'configs/generation_protocol.json').read_bytes()).hexdigest()
    wait_for(RUN.parent / '016/results/final_audit.json', 'qwen016_final_audit')
    assert read(RUN.parent / '016/results/final_audit.json')['verified']
    wait_for(RUN / 'results/gpu_preparation_complete.json', 'target_model_features_and_hessian')
    previous_manifest = RUN.parent / '016/results/manifest.json'
    protected = RUN / 'reference/protected_run016_manifest.json'
    if not protected.exists():
        write(protected, {'sha256':hashlib.sha256(previous_manifest.read_bytes()).hexdigest()})
    from resources import speed_gpu, quality_gpus
    if not (RUN / 'results/profile_complete.json').exists():
        state('kernel_profile', 'waiting')
        with speed_gpu():
            stage('kernel_profile', 'profile_actions.py')
    reference = RUN / 'reference/graph_action_profile.csv'
    if not reference.exists():
        shutil.copy2(RUN / 'results/graph_action_profile.csv', reference)
    baseline = RUN / 'reference/calibration_uniform_dense_bf16.json'
    if not baseline.exists():
        state('allocation_baseline', 'waiting')
        with speed_gpu():
            state('allocation_baseline')
            model = checkpoint('uniform_dense_bf16', 0)
            invoke([sys.executable, RUN / 'code/benchmark.py', '--checkpoint', model,
                '--policy', 'uniform_dense_bf16', '--mode', 'arena_graph', '--label', 'allocation_baseline_bf16',
                '--outputs', '1', '64', '--iterations', '5',
                '--input-file', RUN / 'configs/calibration_speed_inputs.json'], 'allocation_baseline_bf16', 0)
        shutil.copy2(RUN / 'results/measurements/allocation_baseline_bf16.json', baseline)
    if not (RUN / 'results/label_cache_decision.json').exists():
        state('training_cache_gate', 'waiting')
        with quality_gpus(maximum=1, minimum=1) as cards:
            state('training_cache_gate', gpu=cards[0])
            invoke([sys.executable, RUN / 'code/cache_gate.py', '--gpu', str(cards[0])], 'training_cache_gate', cards[0])
    if not (RUN / 'results/training_complete.json').exists():
        stage('training_labels', 'job_queue.py', '--stage', 'training')
    if not (RUN / 'results/quality_model_prefill_decode.json').exists():
        stage('quality_fit', 'fit_quality.py')
    if not (RUN / 'configs/allocation_pool.json').exists():
        stage('frozen_allocations', 'allocate_suite.py')
    if not (RUN / 'results/validation_complete.json').exists():
        stage('validation_quality', 'job_queue.py', '--stage', 'validation')
    if not (RUN / 'results/calibration_complete.json').exists():
        state('speed_calibration', 'waiting')
        with speed_gpu():
            stage('speed_calibration', 'calibration_queue.py')
    if not (RUN / 'configs/selected_policies.json').exists():
        stage('freeze_selection', 'select_suite.py')
    if not (RUN / 'configs/primary_policies.json').exists():
        stage('freeze_primary_tables', 'freeze_primary.py')
    if not (RUN / 'results/confirmation_complete.json').exists():
        stage('confirmation_quality', 'job_queue.py', '--stage', 'confirmation')
    if not (RUN / 'results/formal_complete.json').exists():
        state('formal_speed', 'waiting')
        with speed_gpu():
            stage('formal_speed', 'formal_queue.py')
    stage('nll_speed_report', 'report.py')
    if not (RUN / 'results/full_quality_complete.json').exists():
        stage('full_task_quality', 'job_queue.py', '--stage', 'full_quality')
    stage('core_complete_report', 'report.py', '--full')
    stage('core_audit', 'audit_core.py')
    state('core_table_complete', 'complete',
        scope='Core main table and central controls finished; supplemental speed points remain explicitly separate.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        old = read(RUN / 'results/worker.json')
        write(RUN / 'results/worker.json', {**old, 'status':'failed', 'error':repr(error), 'failed_at_unix':time.time()})
        raise
