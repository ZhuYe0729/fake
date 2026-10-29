"""Quiet same-card calibration after every parallel quality worker has finished."""
import sys

from launch import RUN, checkpoint, cleanup, invoke, read, write


def main():
    pool = read(RUN / 'configs/reanalysis_pool.json')
    assert all((RUN / f'results/validation/{pid}/{s}.json').exists()
               for pid, p in pool['policies'].items() if p['daso_eligible'] for s in p['scenarios'])
    for pid, policy in pool['policies'].items():
        if not policy['daso_eligible']:
            continue
        label = 'calibration_' + pid
        target = RUN / f'results/measurements/{label}.json'
        if target.exists():
            continue
        model = checkpoint(pid, 0)
        outputs = [o for o, s in [(1, 'prefill_only'), (64, 'prefill_decode')] if s in policy['scenarios']]
        invoke([sys.executable, RUN / 'code/benchmark.py', '--checkpoint', model, '--policy', pid,
                '--mode', 'arena_graph', '--label', label, '--outputs', *map(str, outputs),
                '--iterations', '5', '--input-file', RUN / 'configs/calibration_speed_inputs.json'], label, 0)
        cleanup(model, label)
        print('calibration complete', pid, flush=True)
    write(RUN / 'results/calibration_complete.json', {'complete': True,
          'policies': sum(p['daso_eligible'] for p in pool['policies'].values()), 'gpu': 0,
          'scope': 'No own parallel quality computation; fixed calibration seed independent of formal test inputs.'})


if __name__ == '__main__':
    main()
