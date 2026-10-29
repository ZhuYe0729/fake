"""Validate model proposals and freeze deployment choices before confirmation."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from common import RUN, write_json
from ops import b, checkpoint, quality_phase, read, release, state
from resources import lease
from selection import guard, select


def cpu(script, label):
    command = ['numactl', '--all', '--physcpubind=48-55', '--membind=1', sys.executable, str(script)]
    log = RUN / f'logs/{label}_{time.time_ns()}.log'
    write_json(log.with_suffix('.command.json'), {'command': command, 'device': 'CPU',
        'code_sha256': {p.name: b.sha(p) for p in (RUN / 'code').glob('*.py')}})
    start = time.perf_counter()
    with log.open('w') as handle:
        result = subprocess.run(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=handle, stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'), {'seconds': time.perf_counter() - start, 'returncode': result.returncode})
    assert result.returncode == 0, str(log)


def validate():
    pool = read(RUN / 'configs/proposal_pool.json')
    design = read(RUN / 'configs/design.json')
    base = read(RUN / 'results/validation/uniform_dense_bf16/prefill_decode.json')
    rows = []
    for pid in pool['policies']:
        value = read(RUN / f'results/validation/{pid}/prefill_decode.json')
        assert value['data_sha256'] == design['datasets']['validation']['sha256']
        assert len(value['blocks']) == 128 and value['data_role'] == 'validation'
        delta = [x['avg_nll'] - y['avg_nll'] for x, y in zip(value['blocks'], base['blocks'], strict=True)]
        rows.append({'policy': pid, **guard(delta, pid == 'uniform_dense_bf16')})
    write_json(RUN / 'results/validation_candidates.json', rows)
    return rows


def calibration_speed(rows):
    anchors = read(RUN / 'configs/design.json')['anchors']
    pids = [r['policy'] for r in rows if r['guarded_delta_nll'] <= .04 or r['policy'] in anchors]
    for i, pid in enumerate(pids):
        if (RUN / f'results/measurements/calibration_{pid}.json').exists():
            continue
        state('calibration_speed', complete=i, total=len(pids), policy=pid)
        with lease(exclusive=True) as gpu:
            b.benchmark(pid, 'calibration_' + pid, RUN / 'configs/speed_calibration.json', gpu)
    return pids


def freeze(rows, pids):
    assert not list((RUN / 'results/confirmation').glob('*/*.json'))
    design = read(RUN / 'configs/design.json')
    pool = read(RUN / 'configs/proposal_pool.json')
    assert pool['model_choice_sha256'] == b.sha(RUN / 'configs/model_choice.json')
    candidates = []
    for row in rows:
        pid = row['policy']
        if pid in pids:
            median = read(RUN / f'results/measurements/calibration_{pid}.json')['cases'][0]['medians']
            candidates.append({**row, **median})
    choices = []
    for model in design['models']:
        allowed = set(design['anchors']) | set(pool['families'][model])
        for budget in design['budgets']:
            winner, number = select([p for p in candidates if p['policy'] in allowed], budget, 'e2e_ms')
            choices.append({'model': model, 'quality_budget': budget, 'policy': winner['policy'],
                'objective': 'e2e_ms', 'feasible_candidates': number, 'validation_and_calibration': winner})
    write_json(RUN / 'configs/model_selections.json', {'choices': choices,
        'scope': 'Each model uses only its own proposals plus common anchors; calibration comparisons are not all formal test timings.'})
    selected = [c for c in choices if c['model'] == pool['primary_model']]
    write_json(RUN / 'configs/selected_policies.json', {'selected': selected, 'primary_model': pool['primary_model'],
        'frozen_at_unix': time.time(), 'design_sha256': b.sha(RUN / 'configs/design.json'),
        'pool_sha256': b.sha(RUN / 'configs/proposal_pool.json'),
        'registry_sha256': b.sha(RUN / 'configs/policy_registry.json'),
        'validation_sha256': b.sha(RUN / 'results/validation_candidates.json'),
        'calibration_sha256': {p: b.sha(RUN / f'results/measurements/calibration_{p}.json') for p in pids},
        'confirmation_started': False})
    confirm = sorted(set(design['anchors']) | {c['policy'] for c in selected})
    write_json(RUN / 'configs/confirmation_jobs.json', {'jobs': [{'policy': p} for p in confirm]})


def generation_jobs():
    selected = read(RUN / 'configs/selected_policies.json')['selected']
    pids = sorted({p['policy'] for p in selected} | {'uniform_int8', 'uniform_sq_int8', 'phase_sq_int8_w4a16_ours'})
    jobs, inherited = [], []
    for pid in pids:
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        matches = []
        for rid, prefix in [('027', 'full'), ('028', '')]:
            root = RUN.parent / rid
            assert all(b.sha(RUN / 'code' / name) == b.sha(root / 'code' / name) for name in
                ['generate_tasks.py', 'eightbit.py', 'runtime_extensions.py', 'arena_runtime.py', 'graph_bindings.py'])
            for path in (root / 'configs/policies').glob('*.json'):
                if read(path)['method_map'] == mapping:
                    matches.append((root / prefix / 'results/generation' / path.stem, rid))
        for dataset in ['cnn_dm_1000', 'dsum', 'IWSLT']:
            found = next(((p / dataset / 'metrics_0000_end.json', rid) for p, rid in matches
                          if (p / dataset / 'metrics_0000_end.json').exists()), None)
            if found:
                metric, rid = found
                inherited.append({'policy': pid, 'dataset': dataset, 'source': str(metric),
                    'sha256': b.sha(metric), 'source_run': rid, 'metric': read(metric), 'same_map': True})
            else:
                jobs.append({'policy': pid, 'dataset': dataset})
    write_json(RUN / 'configs/generation_jobs.json', {'jobs': jobs, 'reused': inherited,
        'selected_sha256': b.sha(RUN / 'configs/selected_policies.json')})


def main():
    with (RUN / 'results/driver.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(RUN / 'results/design_checks.json')['verified']
        # Calibration was started independently while the remaining CPU
        # orchestration/report code was prepared. Do not launch a second queue.
        while not (RUN / 'results/calibration_complete.json').exists():
            time.sleep(5)
        if not (RUN / 'results/fit_complete.json').exists():
            state('fit')
            cpu(RUN / 'code/fit.py', 'fit_models')
        if not (RUN / 'results/validation_complete.json').exists():
            quality_phase('validation')
        rows = validate()
        if not (RUN / 'configs/selected_policies.json').exists():
            freeze(rows, calibration_speed(rows))
        if not (RUN / 'results/confirmation_complete.json').exists():
            quality_phase('confirmation')
        if not (RUN / 'results/formal_complete.json').exists():
            comparisons = read(RUN / 'configs/model_selections.json')['choices']
            b.CONTROLS = sorted(set(b.CONTROLS) | {p['policy'] for p in comparisons})
            b.formal()
        if not (RUN / 'configs/generation_jobs.json').exists():
            generation_jobs()
        if not (RUN / 'results/generation_complete.json').exists():
            quality_phase('generation')
        for path in (RUN / 'data/checkpoints').glob('*/model.safetensors'):
            release(path.parent, 'final_' + path.parent.name)
        state('report')
        cpu(RUN / 'code/report.py', 'report')
        write_json(RUN / 'results/worker.json', {'status': 'complete', 'stage': 'audited'})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(RUN / f'results/failure_{os.getpid()}_{time.time_ns()}.json', {'error': repr(error)})
        raise
