"""Select only the full-eight-action extension using fresh validation and calibration."""
import hashlib
import math
import statistics

from common import RUN, write_json
from launch import read


def main():
    assert not (RUN / 'configs/selected_policies.json').exists()
    assert not list((RUN / 'results/confirmation').glob('*/*.json'))
    pool = read(RUN / 'configs/allocation_pool.json')
    rules = read(RUN / 'configs/frozen_design.json')
    candidates = []
    for scene in ['prefill_only', 'prefill_decode']:
        baseline = read(RUN / f'results/validation/uniform_dense_bf16/{scene}.json')
        for pid, spec in pool['policies'].items():
            if scene not in spec['selection_scenarios']:
                continue
            quality = read(RUN / f'results/validation/{pid}/{scene}.json')
            differences = [a['avg_nll']-b['avg_nll'] for a,b in zip(quality['blocks'], baseline['blocks'], strict=True)]
            assert len(differences) == 32
            mean = statistics.mean(differences)
            se = statistics.stdev(differences)/math.sqrt(32)
            guard = 0. if pid == 'uniform_dense_bf16' else mean+2*se+.005
            speed = read(RUN / f'results/measurements/calibration_{pid}.json')
            case = next(c for c in speed['cases'] if c['output'] == (1 if scene == 'prefill_only' else 64))
            candidates.append({'policy': pid, 'scenario': scene, 'validation_delta_nll': mean,
                'paired_se': se, 'guarded_delta_nll': guard,
                'calibration_e2e_ms': case['medians']['e2e_ms'],
                'calibration_cycle_ms': case['medians']['cycle_ms']})
    selected = []
    for scene in ['prefill_only', 'prefill_decode']:
        for budget in rules['quality_budgets']:
            feasible = [p for p in candidates if p['scenario'] == scene and p['guarded_delta_nll'] <= budget]
            winner = min(feasible, key=lambda p:(p['calibration_e2e_ms'], p['policy']))
            selected.append({'label': f'daso8_verified_{scene}_q{budget:g}', 'scenario': scene,
                'quality_budget': budget, 'policy': winner['policy'], 'validation': winner,
                'feasible_candidates': len(feasible)})
    write_json(RUN / 'results/validation_candidates.json', candidates)
    write_json(RUN / 'configs/selected_policies.json', {'selected': selected,
        'pool_sha256': hashlib.sha256((RUN / 'configs/allocation_pool.json').read_bytes()).hexdigest(),
        'design_sha256': hashlib.sha256((RUN / 'configs/frozen_design.json').read_bytes()).hexdigest(),
        'scope': 'DASO-8 extension only. Raw matched-cap allocations remain fixed, not chosen by validation or confirmation. No HAWQ-only map is added to the selectable pool; shared hardware/uniform anchors are explicit.'})
    print('Frozen four validation-selected points before confirmation.', flush=True)


if __name__ == '__main__':
    main()
