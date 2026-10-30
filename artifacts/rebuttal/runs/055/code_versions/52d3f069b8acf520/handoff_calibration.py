"""Finish the active label, then retire only the old run055 queue parent."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from calibration_queue import digest
from common import RUN, write_json


def command(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return [value.decode() for value in path.read_bytes().split(b'\0') if value] if path.exists() else []


def children(pid):
    path = Path(f'/proc/{pid}/task/{pid}/children')
    return list(map(int, path.read_text().split())) if path.exists() else []


def state(pid):
    raw = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
    return raw[0], int(raw[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=int, required=True)
    args = parser.parse_args()
    parent = args.parent
    assert str(RUN / 'code/calibration_queue.py') in command(parent) or 'artifacts/rebuttal/runs/055/code/calibration_queue.py' in command(parent)
    assert '--gpu' in command(parent) and command(parent)[command(parent).index('--gpu') + 1] == '1'
    paused = False
    try:
        while True:
            candidates = [pid for pid in children(parent) if str(RUN / 'code/experiment.py') in command(pid)]
            if not candidates:
                time.sleep(1); continue
            os.kill(parent, signal.SIGSTOP)
            paused = True
            while state(parent)[0] != 'T':
                time.sleep(.05)
            candidates = [pid for pid in children(parent) if str(RUN / 'code/experiment.py') in command(pid)]
            if candidates:
                child, = candidates
                argv = command(child)
                break
            os.kill(parent, signal.SIGCONT); paused = False
        policy = argv[argv.index('--policy') + 1]
        label = argv[argv.index('--label') + 1]
        record_path = RUN / f'results/control/queue_handoff_{time.time_ns()}.json'
        record = {'parent': parent, 'child': child, 'argv': argv, 'policy': policy, 'label': label,
                  'scope': 'Queue parent paused; active GPU label allowed to finish normally.'}
        write_json(record_path, record)
        while state(child)[0] != 'Z':
            time.sleep(1)
        assert state(child)[1] == 0, 'Active label failed; resume original parent to handle its exception.'
        measurement = RUN / f'results/measurements/{label}.json'
        body = json.loads(measurement.read_text())
        assert body['complete'] and all(value['exact_equal'] for value in body['weight_audit'].values())
        design = json.loads((RUN / 'configs/design.json').read_text())
        results = []
        for case in design['workloads']:
            path = RUN / f"results/quality/calibration/{policy}/{case['label']}.json"
            value = json.loads(path.read_text())
            assert value['complete'] and len(value['blocks']) == 100
            assert value['sample_sha256'] == design['quality_sha256']['calibration']
            results.append({'path': str(path), 'sha256': digest(path)})
        evidence = RUN / f'results/export_identity/{label}'
        identity = json.loads((evidence / 'identity.json').read_text())
        write_json(RUN / f'results/calibration_complete/{policy}.json', {'complete': True,
            'policy': policy, 'gpu': 1, 'export_identity': str(evidence / 'identity.json'),
            'quality': results, 'handoff': str(record_path), 'child_exit_code': 0})
        if identity['temporary']:
            binary = Path(identity['checkpoint']) / 'model.safetensors'
            binary.unlink()
            write_json(evidence / 'temporary_weight_cleanup.json', {'path': str(binary),
                'bytes': identity['binary_bytes'], 'sha256': identity['binary_sha256'],
                'reason': 'Successful label finished normally before queue-parent handoff; only reproducible weight binary removed.'})
        os.kill(parent, signal.SIGTERM); os.kill(parent, signal.SIGCONT)
        paused = False
        record.update(complete=True, active_label_finished_normally=True,
                      measurement_sha256=digest(measurement), queue_parent_retired=True)
        write_json(record_path, record)
        print('Retired old queue parent after complete label:', policy, flush=True)
    finally:
        if paused:
            os.kill(parent, signal.SIGCONT)


if __name__ == '__main__':
    main()
