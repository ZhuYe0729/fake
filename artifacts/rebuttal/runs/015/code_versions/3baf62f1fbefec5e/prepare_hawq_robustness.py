"""Prespecify trace-sign sensitivity at the central budget, without moving main rows."""
import hashlib
import shutil

from common import RUN, write_json
from launch import read
from prepare_hawq_controls import main as freeze


def main():
    destination = RUN / 'configs/hawq_robustness.json'
    assert not destination.exists()
    # 35/252 individual trace estimates are negative. Reuse run009's two
    # prespecified treatments rather than choose a treatment from new test scores.
    points = []
    for transform in ['clip_zero', 'absolute']:
        freeze(transform, (.80,))
        points.extend(read(RUN / f'configs/hawq_{transform}_controls.json')['points'])
    jobs = {}
    design = read(RUN / 'configs/design.json')
    for p in points:
        pid, scenario = p['policy_id'], p['scenario']
        jobs.setdefault(pid, {'policy': pid, 'scenarios': []})['scenarios'].append(scenario)
        spec = design['policies'].setdefault(pid, {'outputs': []})
        output = 1 if scenario == 'prefill_only' else 64
        if output not in spec['outputs']:
            spec['outputs'].append(output)
            spec['outputs'].sort()
    other = sorted(set(design['policies']) - {'uniform_dense_bf16'})
    design['rounds'] = [{'round': i, 'order': ['uniform_dense_bf16', *order]} for i, order in
        enumerate([other, other[::-1], other[7:] + other[:7]])]
    shutil.copy2(RUN / 'configs/design.json', RUN / 'reference/pre_hessian_robustness_design.json')
    write_json(RUN / 'configs/design.json', design)
    write_json(destination, {'points': points, 'jobs': list(jobs.values()),
        'selection_sha256': hashlib.sha256((RUN / 'configs/selected_reanalysis.json').read_bytes()).hexdigest(),
        'scope': 'Baseline sensitivity only, central .80 budget, signed main rows and DASO selection unchanged. Motivated by 35/252 negative individual traces before inspection of new HAWQ test quality; no new score was used to choose these variants. All points receive test NLL and three formal speed campaigns, without test-based filtering.'})
    print('Robustness controls', len(points), 'policy jobs', len(jobs),
          'total formal policies', len(design['policies']), flush=True)


if __name__ == '__main__':
    main()
