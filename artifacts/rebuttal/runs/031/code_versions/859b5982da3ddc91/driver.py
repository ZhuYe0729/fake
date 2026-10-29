"""Small full-space comparison; freeze choices before independent confirmation."""
import fcntl
import subprocess
import sys
import time

from common import RUN, write_json
from ops import b, quality_phase, read, release, state
from resources import lease
from selection import guard, select


def validate():
    design = read(RUN / 'configs/design.json')
    base = read(RUN / 'results/validation/uniform_dense_bf16/prefill_decode.json')
    rows = []
    for pid in read(RUN / 'configs/proposal_pool.json')['policies']:
        value = read(RUN / f'results/validation/{pid}/prefill_decode.json')
        assert value['data_sha256'] == design['datasets']['validation']['sha256']
        assert len(value['blocks']) == 128 and value['data_role'] == 'validation'
        delta = [a['avg_nll'] - z['avg_nll'] for a, z in zip(value['blocks'], base['blocks'], strict=True)]
        rows.append({'policy': pid, **guard(delta, pid == 'uniform_dense_bf16')})
    write_json(RUN / 'results/validation_candidates.json', rows)
    return rows


def freeze(rows):
    assert not list((RUN / 'results/confirmation').glob('*/*.json'))
    design = read(RUN / 'configs/design.json')
    pool = read(RUN / 'configs/proposal_pool.json')
    eligible = [r for r in rows if r['guarded_delta_nll'] <= .04 or r['policy'] in design['anchors']]
    candidates = []
    for i, row in enumerate(eligible):
        pid = row['policy']
        state('calibration_speed', complete=i, total=len(eligible), policy=pid)
        with lease(exclusive=True) as gpu:
            b.benchmark(pid, 'calibration_' + pid, RUN / 'configs/speed_calibration.json', gpu)
        median = read(RUN / f'results/measurements/calibration_{pid}.json')['cases'][0]['medians']
        candidates.append({**row, **median})
    choices = []
    for model in design['models']:
        allowed = set(design['anchors']) | set(pool['families'][model])
        for budget in design['budgets']:
            winner, number = select([r for r in candidates if r['policy'] in allowed], budget, 'e2e_ms')
            choices.append({'model': model, 'quality_budget': budget, 'policy': winner['policy'],
                'objective': 'e2e_ms', 'feasible_candidates': number, 'validation_and_calibration': winner})
    write_json(RUN / 'configs/model_selections.json', {'choices': choices})
    write_json(RUN / 'configs/selected_policies.json', {'selected': choices, 'primary_model': 'signed_residual',
        'frozen_at_unix': time.time(), 'design_sha256': b.sha(RUN / 'configs/design.json'),
        'pool_sha256': b.sha(RUN / 'configs/proposal_pool.json'),
        'registry_sha256': b.sha(RUN / 'configs/policy_registry.json'),
        'validation_sha256': b.sha(RUN / 'results/validation_candidates.json'),
        'calibration_sha256': {r['policy']: b.sha(RUN / f'results/measurements/calibration_{r["policy"]}.json') for r in eligible}})
    confirm = sorted({'uniform_dense_bf16', 'uniform_int8', 'uniform_sq_int8', 'group_3ee5'} | {c['policy'] for c in choices})
    write_json(RUN / 'configs/confirmation_jobs.json', {'jobs': [{'policy': p} for p in confirm]})


def main():
    with (RUN / 'results/driver.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(RUN / 'results/design_checks.json')['verified']
        assert read(RUN / 'results/solver_gate.json')['verified']
        if not (RUN / 'results/validation_complete.json').exists():
            quality_phase('validation')
        rows = validate()
        if not (RUN / 'configs/selected_policies.json').exists():
            freeze(rows)
        if not (RUN / 'results/confirmation_complete.json').exists():
            quality_phase('confirmation')
        b.CONTROLS = ['uniform_dense_bf16', 'uniform_int8', 'uniform_sq_int8', 'group_3ee5']
        if not (RUN / 'results/formal_complete.json').exists():
            b.formal()
        for path in (RUN / 'data/checkpoints').glob('*/model.safetensors'):
            release(path.parent, 'final_' + path.parent.name)
        state('report')
        subprocess.run([sys.executable, RUN / 'code/report.py'], check=True)
        write_json(RUN / 'results/worker.json', {'status': 'complete', 'stage': 'audited'})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(RUN / f'results/failure_{time.time_ns()}.json', {'error': repr(error)})
        raise
