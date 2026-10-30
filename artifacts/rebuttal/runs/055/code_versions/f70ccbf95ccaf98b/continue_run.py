"""Advance the authorized run after both calibration workers finish."""
import fcntl
import json
import os
import subprocess
import sys
import time

from common import RUN, write_json


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    with (RUN / 'continuation.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((RUN / 'configs/calibration_manifest.json').read_text())
        while True:
            complete = sum((RUN / f"results/calibration_complete/{row['policy_id']}.json").exists()
                           for row in manifest['entries'])
            prompt = RUN / 'results/prompt_label_reuse_check.json'
            write_json(RUN / 'results/continuation_status.json', {'stage': 'waiting_calibration',
                'complete_policies': complete, 'total_policies': len(manifest['entries']), 'unix': time.time()})
            if complete == len(manifest['entries']) and prompt.exists():
                assert json.loads(prompt.read_text())['complete']
                break
            time.sleep(10)
        stages = [('fit', 'fit_quality.py', [], False),
            ('prepare_validation', 'select_points.py', ['prepare'], False),
            ('validate_daso', 'select_points.py', ['evaluate'], True),
            ('select_daso', 'select_points.py', ['select'], False),
            ('allocate_hawq', 'allocate_hawq.py', [], False),
            ('validate_hawq', 'close_quality.py', ['hawq-validation'], True),
            ('freeze', 'freeze_formal.py', [], False),
            ('test_nll', 'close_quality.py', ['test'], True),
            ('speed', 'formal_speed.py', [], False),
            ('downstream', 'downstream.py', [], False),
            ('report', 'final_report.py', [], False)]
        for stage, script, args, needs_gpu_lock in stages:
            target = RUN / f'results/continuation_stages/{stage}.json'
            if target.exists():
                assert json.loads(target.read_text())['complete']
                continue
            label = f'continuation_{stage}_{time.time_ns()}'
            command = [sys.executable, RUN / 'code' / script, *args]
            if needs_gpu_lock:
                command = ['flock', RUN / 'gpu0.lock', 'numactl', '--physcpubind=0-7', '--membind=0', *command]
            write_json(RUN / 'results/continuation_status.json', {'stage': stage, 'unix': time.time(),
                'command': [str(value) for value in command], 'log': str(RUN / f'logs/{label}.log')})
            started = time.time()
            with (RUN / f'logs/{label}.log').open('w') as stream:
                result = subprocess.run([str(value) for value in command], stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(RUN / 'results/continuation_status.json', {'stage': stage, 'status': 'failed',
                    'returncode': result.returncode, 'log': str(RUN / f'logs/{label}.log'), 'unix': time.time()})
                raise RuntimeError(f'{stage} failed; evidence retained in {label}.log')
            write_json(target, {'complete': True, 'stage': stage, 'label': label,
                'started_unix': started, 'finished_unix': time.time(), 'elapsed_seconds': time.time() - started})
            print('complete stage', stage, flush=True)
        write_json(RUN / 'results/continuation_status.json', {'stage': 'complete', 'unix': time.time()})


if __name__ == '__main__':
    main()
