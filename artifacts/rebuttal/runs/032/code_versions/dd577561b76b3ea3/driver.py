"""Evaluate every prespecified budget/weight point, retaining all results."""
import fcntl
import subprocess
import sys
import time

from common import RUN, write_json
from ops import b, quality_phase, read, release, state


def main():
    with (RUN / 'results/driver.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(RUN / 'results/design_checks.json')['verified']
        for stage in ['evaluation', 'generation']:
            if not (RUN / f'results/{stage}_complete.json').exists():
                quality_phase(stage)
        b.CONTROLS = read(RUN / 'configs/design.json')['anchors']
        if not (RUN / 'results/formal_complete.json').exists():
            b.formal()
        if not (RUN / 'results/mc_complete.json').exists():
            quality_phase('mc')
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
