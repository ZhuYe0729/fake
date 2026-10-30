"""Three fixed serial campaigns, with BF16 brackets and resource telemetry."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from common import RUN, write_json
import policy_runtime
from resources import descendants, idle, snapshot

OWNED = {os.getpid()}


class ResourceInterference(RuntimeError):
    pass


def run_owned(command, label):
    command = [str(value) for value in command]
    log = RUN / f'logs/{label}.log'
    write_json(log.with_suffix('.command.json'), {'argv': command, 'gpu': 0,
        'cpu': '0-7', 'numa': 0, 'time_ns': time.time_ns()})
    with log.open('w') as output, (RUN / f'logs/{label}.resources.jsonl').open('w') as telemetry:
        process = subprocess.Popen(['numactl', '--physcpubind=0-7', '--membind=0', *command],
                                   stdout=output, stderr=subprocess.STDOUT)
        OWNED.clear(); OWNED.update({os.getpid(), process.pid})
        previous, suspicious = None, 0
        try:
            while process.poll() is None:
                OWNED.update(descendants(process.pid))
                rows = snapshot()
                telemetry.write(json.dumps({'unix': time.time(), 'owned_pids': sorted(OWNED), 'gpus': rows}) + '\n')
                telemetry.flush()
                foreign = [item for row in rows for item in row['processes'] if item['pid'] not in OWNED]
                other_busy = any(row['index'] != 0 and row['allocated_bytes'] >= 128 * 1024**2 for row in rows)
                current = next(row for row in rows if row['index'] == 0)
                own_bytes = sum(item['used_bytes'] for item in current['processes'] if item['pid'] in OWNED)
                gap = current['allocated_bytes'] - own_bytes
                stable = previous is not None and abs(previous['allocated_bytes'] - current['allocated_bytes']) < 32 * 1024**2
                suspicious = suspicious + 1 if stable and own_bytes > 0 and gap > 128 * 1024**2 else 0
                if foreign or other_busy or suspicious >= 2:
                    raise ResourceInterference(json.dumps({'foreign': foreign, 'other_busy': other_busy, 'gap': gap}))
                previous = current
                time.sleep(1)
            assert process.returncode == 0, f'{label}: exit {process.returncode}; see {log}'
        except BaseException:
            if process.poll() is None:
                for pid in descendants(process.pid):
                    try: os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise


def wait_idle(label):
    since = None
    with (RUN / f'logs/{label}.idle.jsonl').open('w') as stream:
        while True:
            now = time.monotonic()
            rows = snapshot()
            stream.write(json.dumps({'unix': time.time(), 'gpus': rows}) + '\n'); stream.flush()
            since = (since if since is not None else now) if idle(rows) else None
            if since is not None and now - since >= 90:
                return
            time.sleep(10)


def measure(policy, scenario, label, round_index):
    with policy_runtime.checkpoint(policy, label) as root:
        command = [sys.executable, RUN / 'code/experiment.py', '--checkpoint', root, '--policy', policy,
            '--mode', 'speed', '--tier', '1', '--round', str(round_index), '--label', label]
        if scenario is not None:
            command += ['--scenario', scenario]
        run_owned(command, label)
    path = RUN / f'results/measurements/{label}.json'
    body = json.loads(path.read_text())
    assert body['complete'] and all(value['exact_equal'] for value in body['weight_audit'].values())
    assert len(body['cases']) == (2 if scenario is None else 1)
    for case in body['cases']:
        assert scenario is None or case['case']['scenario'] == scenario
        assert len(case['samples']) == 15 and sum(not sample['warmup'] for sample in case['samples']) == 10
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'policy': policy, 'scenario': scenario}


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    formal_path = RUN / 'configs/formal_points.json'
    formal = json.loads(formal_path.read_text())
    assert json.loads((RUN / 'results/test_complete.json').read_text())['complete']
    assert json.loads((RUN / 'results/hessian_trace_p16_b4_s256.json').read_text())['complete']
    manifest = json.loads((RUN / 'configs/calibration_manifest.json').read_text())
    assert all((RUN / f"results/calibration_complete/{entry['policy_id']}.json").exists() for entry in manifest['entries'])
    baseline = 'uniform_dense_bf16'
    baseline_map = json.loads((RUN / f'configs/policies/{baseline}.json').read_text())['method_map']
    baseline_signature = hashlib.sha256(json.dumps(baseline_map, sort_keys=True).encode()).hexdigest()
    cases = [case for case in formal['unique_cases'] if case['map_signature'] != baseline_signature]
    cases.sort(key=lambda case: (case['scenario'], case['case_key']))
    policy_runtime.invoke = run_owned
    with (RUN / 'gpu0.lock').open('a') as lock0, (RUN / 'gpu1.lock').open('a') as lock1:
        fcntl.flock(lock0, fcntl.LOCK_EX); fcntl.flock(lock1, fcntl.LOCK_EX)
        for round_index in range(3):
            destination = RUN / f'results/speed_rounds/round{round_index}.json'
            if destination.exists():
                record = json.loads(destination.read_text()); assert record['complete']
                assert record['formal_points_sha256'] == hashlib.sha256(formal_path.read_bytes()).hexdigest()
                continue
            while True:
                attempt = f'speed_r{round_index}_{time.time_ns()}'
                wait_idle(attempt)
                records = []
                try:
                    records.append(measure(baseline, None, attempt + '_bf16_start', round_index))
                    for case in reversed(cases) if round_index == 1 else cases:
                        records.append(measure(case['policy_id'], case['scenario'], attempt + '_' + case['case_key'], round_index))
                    records.append(measure(baseline, None, attempt + '_bf16_end', round_index))
                except ResourceInterference as error:
                    relocated = []
                    for root in Path('/dev/shm/daso_rebuttal_055').glob(attempt + '*'):
                        destination_root = RUN / 'exports/interfered' / root.name
                        shutil.copytree(root, destination_root)
                        for source in root.iterdir():
                            if source.is_file() and (source.suffix == '.safetensors' or source.name.startswith('.tmp')):
                                copy = destination_root / source.name
                                assert policy_runtime.digest(source) == policy_runtime.digest(copy)
                                relocated.append({'from': str(source), 'to': str(copy), 'sha256': policy_runtime.digest(copy)})
                                source.unlink()
                    write_json(RUN / f'results/interfered_speed/{attempt}.json', {'complete': False,
                        'reason': str(error), 'completed_records': records, 'relocated_weight_files': relocated,
                        'scope': 'Entire matched round repeated; all attempts retained. Failed temporary weights moved losslessly off RAM.'})
                    continue
                record = {'complete': True, 'round': round_index, 'records': records,
                    'formal_points_sha256': hashlib.sha256(formal_path.read_bytes()).hexdigest(),
                    'scope': 'Same GPU0/NUMA0/CPU0-7; 90s idle admission; no own jobs on GPU1; five warmups and ten timed iterations; BF16 brackets.'}
                write_json(destination, record); break
        write_json(RUN / 'results/formal_speed_complete.json', {'complete': True, 'rounds': 3,
            'round_files': [str(RUN / f'results/speed_rounds/round{index}.json') for index in range(3)],
            'scope': 'All complete rounds retained regardless of ordering or variance; >5% variation is flagged in reporting, not used to delete rounds.'})


if __name__ == '__main__':
    main()
