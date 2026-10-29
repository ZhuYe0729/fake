"""Evaluate the frozen repair pool, select, then open confirmation and tasks."""
import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import MODEL, RUN, WRAPPER, write_json
from selection import guard, select

ROOT = RUN.parents[1] / 'night151'
CONTROLS = ['uniform_dense_bf16', 'uniform_int8', 'uniform_sq_int8',
            'phase_dense_nvfp4_sq_int8', 'phase_sq_int8_w4a16_ours']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(stage, **details):
    write_json(RUN / 'results/worker.json', {'stage': stage, 'status': 'running', **details})


def invoke(argv, label, gpu):
    command = ['numactl', '--all', f'--physcpubind={gpu * 8}-{gpu * 8 + 7}',
               f'--membind={0 if gpu < 4 else 1}', *map(str, argv)]
    log = RUN / f'logs/{label}_{time.time_ns()}.log'
    hashes = {p.name: sha(p) for p in (RUN / 'code').glob('*.py')}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = RUN / 'code_versions' / version
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in (RUN / 'code').glob('*.py'):
        if not (snapshot / p.name).exists():
            shutil.copy2(p, snapshot / p.name)
    write_json(log.with_suffix('.command.json'), {'command': command, 'gpu': gpu,
        'design_sha256': sha(RUN / 'configs/design.json'), 'code_sha256': hashes, 'code_version': version})
    start = time.perf_counter()
    with log.open('w') as handle:
        result = subprocess.run(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                stdout=handle, stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'), {'seconds': time.perf_counter() - start, 'returncode': result.returncode})
    assert result.returncode == 0, str(log)


def checkpoint(pid, gpu):
    mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
    spec = read(RUN / 'configs/design.json')['policies'][pid]
    assert sha(RUN / f'configs/policies/{pid}.json') == spec['policy_sha256']
    sources = list((RUN.parent / '027/data/checkpoints').glob('*'))
    sources.append(Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_dense_bf16'))
    for source in sources:
        config = source / 'config.json'
        if not config.exists() or not (source / 'model.safetensors').exists():
            continue
        effective = read(config).get('quantization_config', {}).get('method_map')
        if effective == mapping:
            write_json(RUN / f'results/checkpoint_reuse/{pid}.json', {'source': str(source),
                'config_sha256': sha(config), 'signature': spec['signature'], 'effective_map_equal': True})
            return source
    model = RUN / 'data/checkpoints' / pid
    if not (model / 'model.safetensors').exists():
        with (ROOT / 'export.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            assert shutil.disk_usage(RUN).free > 64 * 1024**3
            if model.exists():
                model.rename(model.with_name(pid + '_previous_' + str(time.time_ns())))
            invoke([sys.executable, RUN / 'code/export_model.py', '--model-path', MODEL,
                    '--policy-json', RUN / f'configs/policies/{pid}.json', '--output-dir', model,
                    '--cutlass-wrapper-path', WRAPPER], 'export_' + pid, gpu)
    assert read(model / 'config.json')['quantization_config']['method_map'] == mapping
    return model


def cleanup(model, label):
    if model.parent != RUN / 'data/checkpoints':
        return
    path = model / 'model.safetensors'
    if path.exists():
        assert not path.is_symlink()
        record = {'path': str(path), 'bytes': path.stat().st_size,
            'reason': 'Only own reproducible exported weights removed; metadata/code/logs/results retained.'}
        path.unlink()
        write_json(RUN / f'results/cleanup/{label}_{time.time_ns()}.json', record)


def quality_worker(stage):
    from resources import lease
    design = read(RUN / 'configs/design.json')
    if stage == 'validation':
        jobs = [{'policy': p} for p in design['policies']]
    elif stage == 'confirmation':
        selected = read(RUN / 'configs/selected_policies.json')
        jobs = [{'policy': p} for p in sorted(set(CONTROLS) | {x['policy'] for x in selected['selected']})]
    else:
        jobs = read(RUN / 'configs/generation_jobs.json')['jobs']
    for job in jobs:
        pid = job['policy']
        label = pid + ('_' + job['dataset'] if stage == 'generation' else '')
        marker = RUN / f'results/completed/{stage}/{label}.json'
        if marker.exists():
            continue
        # Full datasets for one owned checkpoint run serially: another task
        # must not clean its weights while a sibling is starting its loader.
        # The archived INT8 checkpoint is immutable and never cleaned here;
        # its three full datasets can safely use one GPU each.
        parallel_int8 = stage == 'generation' and pid == 'uniform_int8'
        lock = RUN / f'results/locks/{stage}_{label if parallel_int8 else pid}.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a') as claim:
            try:
                fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if marker.exists():
                continue
            with lease() as gpu:
                model = checkpoint(pid, gpu)
                if parallel_int8:
                    assert model == RUN.parent / '027/data/checkpoints/uniform_int8'
                if stage == 'generation':
                    argv = [sys.executable, RUN / 'code/generate_tasks.py', '--checkpoint', model,
                            '--policy', pid, '--dataset', job['dataset']]
                else:
                    argv = [sys.executable, RUN / 'code/score_nll.py', '--checkpoint', model,
                            '--policy', pid, '--split', stage, '--scenarios', 'prefill_decode']
                invoke(argv, stage + '_' + label, gpu)
                write_json(marker, {'complete': True, **job, 'gpu': gpu})
                cleanup(model, stage + '_' + label)


def quality_phase(stage, total):
    children = []
    for i in range(3):
        log = (RUN / f'logs/{stage}_worker{i}_{time.time_ns()}.log').open('w')
        child = subprocess.Popen([sys.executable, __file__, '--worker', stage], stdout=log, stderr=subprocess.STDOUT)
        children.append((child, log))
    while any(child.poll() is None for child, _ in children):
        state(stage, complete=len(list((RUN / f'results/completed/{stage}').glob('*.json'))), total=total,
              workers=[{'pid': child.pid, 'returncode': child.poll()} for child, _ in children])
        time.sleep(10)
    codes = [child.returncode for child, _ in children]
    for _, log in children:
        log.close()
    assert codes == [0, 0, 0], codes
    assert len(list((RUN / f'results/completed/{stage}').glob('*.json'))) == total
    write_json(RUN / f'results/{stage}_complete.json', {'complete': True, 'jobs': total})


def validation():
    baseline = read(RUN / 'results/validation/uniform_dense_bf16/prefill_decode.json')
    rows = []
    for pid, spec in read(RUN / 'configs/design.json')['policies'].items():
        score = read(RUN / f'results/validation/{pid}/prefill_decode.json')
        assert len(score['blocks']) == 128 and score['data_sha256'] == baseline['data_sha256']
        differences = [a['avg_nll'] - b['avg_nll'] for a, b in zip(score['blocks'], baseline['blocks'], strict=True)]
        row = {'policy': pid, **spec, **guard(differences, pid == 'uniform_dense_bf16'),
            'sample_size_trace': {str(n): guard(differences[:n], pid == 'uniform_dense_bf16') for n in [32, 64, 128]}}
        rows.append(row)
    write_json(RUN / 'results/validation_candidates.json', rows)
    return rows


def benchmark(pid, label, inputs, gpu, audit=False):
    target = RUN / f'results/measurements/{label}.json'
    if target.exists():
        return
    model = checkpoint(pid, gpu)
    command = [sys.executable, RUN / 'code/benchmark.py', '--checkpoint', model, '--policy', pid,
        '--mode', 'arena_graph', '--label', label, '--outputs', '64', '--iterations', '5', '--input-file', inputs]
    if audit:
        command += ['--audit-weights']
    invoke(command, label, gpu)
    cleanup(model, label)


def calibration(rows):
    from resources import lease
    pids = [r['policy'] for r in rows if r['guarded_delta_nll'] <= .04 or r['policy'] in CONTROLS]
    write_json(RUN / 'configs/calibration_jobs.json', {'policies': pids,
        'rule': 'All guard-feasible at largest frozen budget, plus five fixed controls; no proxy-only rejection.'})
    for index, pid in enumerate(pids):
        if (RUN / f'results/measurements/calibration_{pid}.json').exists():
            continue
        state('calibration_speed', complete=index, total=len(pids), policy=pid)
        with lease(exclusive=True) as gpu:
            benchmark(pid, 'calibration_' + pid, RUN / 'configs/speed_calibration.json', gpu)
    return pids


def freeze_selection(rows, pids):
    assert not list((RUN / 'results/confirmation').glob('*/*.json'))
    candidates = []
    for row in rows:
        pid = row['policy']
        if pid not in pids:
            continue
        speed = read(RUN / f'results/measurements/calibration_{pid}.json')['cases'][0]['medians']
        candidates.append({**row, **speed})
    selected = []
    for budget in [.01, .04]:
        for metric in ['e2e_ms', 'cycle_ms']:
            winner, feasible = select(candidates, budget, metric)
            selected.append({'quality_budget': budget, 'objective': metric, 'policy': winner['policy'],
                             'feasible_candidates': feasible, 'validation_and_calibration': winner})
    write_json(RUN / 'configs/selected_policies.json', {'selected': selected, 'frozen_at_unix': time.time(),
        'design_sha256': sha(RUN / 'configs/design.json'),
        'validation_sha256': sha(RUN / 'results/validation_candidates.json'),
        'calibration_hashes': {p: sha(RUN / f'results/measurements/calibration_{p}.json') for p in pids},
        'confirmation_started': False, 'note': 'Only fixed 128-window validation and calibration timings choose policies; E2E primary and cycle secondary were both declared before labels.'})


def formal():
    from resources import lease
    selected = read(RUN / 'configs/selected_policies.json')['selected']
    pids = sorted(set(CONTROLS) | {s['policy'] for s in selected})
    path = RUN / 'configs/formal_rounds.json'
    if not path.exists():
        rounds = []
        for i in range(3):
            middle = [p for p in pids if p != 'uniform_dense_bf16']
            random.Random(15202817 + i).shuffle(middle)
            rounds.append(['uniform_dense_bf16', *middle, 'uniform_dense_bf16_bracket_end'])
        write_json(path, {'rounds': rounds, 'selected_sha256': sha(RUN / 'configs/selected_policies.json')})
    rounds = read(path)['rounds']
    for campaign, order in enumerate(rounds):
        for index, label in enumerate(order):
            target = f'formal_round{campaign}_{label}'
            if (RUN / f'results/measurements/{target}.json').exists():
                continue
            pid = 'uniform_dense_bf16' if label.endswith('_bracket_end') else label
            state('formal_speed', campaign=campaign, complete=index, total=len(order), policy=pid)
            with lease(exclusive=True) as gpu:
                benchmark(pid, target, RUN / 'configs/speed_formal.json', gpu, audit=True)
    write_json(RUN / 'results/formal_complete.json', {'complete': True, 'campaigns': 3})


def prepare_generation():
    selected = read(RUN / 'configs/selected_policies.json')['selected']
    pids = {p['policy'] for p in selected if p['objective'] == 'e2e_ms'} | {'uniform_int8', 'uniform_sq_int8'}
    source = RUN.parent / '027'
    reused, jobs = [], []
    for pid in sorted(pids):
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        match = next((p.stem for p in (source / 'configs/policies').glob('*.json')
                      if read(p)['method_map'] == mapping), None)
        for dataset in ['cnn_dm_1000', 'dsum', 'IWSLT']:
            metric = source / f'full/results/generation/{match}/{dataset}/metrics_0000_end.json'
            if match is not None and metric.exists():
                reused.append({'policy': pid, 'dataset': dataset, 'source': str(metric), 'sha256': sha(metric),
                    'source_policy': match, 'same_map': True, 'metric': read(metric)})
            else:
                jobs.append({'policy': pid, 'dataset': dataset})
    write_json(RUN / 'configs/generation_jobs.json', {'jobs': jobs, 'reused': reused,
        'selected_sha256': sha(RUN / 'configs/selected_policies.json'),
        'scope': 'Full previously used benchmark inputs; these descriptive scores never choose or repair a map.'})
    return len(jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['validation', 'confirmation', 'generation'])
    args = parser.parse_args()
    if args.worker:
        quality_worker(args.worker)
        return
    with (RUN / 'results/driver.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(RUN / 'results/design_checks.json')['verified']
        if not (RUN / 'results/validation_complete.json').exists():
            quality_phase('validation', 24)
        rows = validation()
        if not (RUN / 'configs/selected_policies.json').exists():
            pids = calibration(rows)
            freeze_selection(rows, pids)
        count = len(set(CONTROLS) | {x['policy'] for x in read(RUN / 'configs/selected_policies.json')['selected']})
        if not (RUN / 'results/confirmation_complete.json').exists():
            quality_phase('confirmation', count)
        if not (RUN / 'results/formal_complete.json').exists():
            formal()
        count = prepare_generation()
        if not (RUN / 'results/generation_complete.json').exists():
            quality_phase('generation', count)
        state('report')
        subprocess.run([sys.executable, RUN / 'code/report.py'], check=True)
        write_json(RUN / 'results/worker.json', {'status': 'complete', 'stage': 'audited'})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(RUN / f'results/failure_{os.getpid()}_{time.time_ns()}.json', {'error': repr(error)})
        raise
