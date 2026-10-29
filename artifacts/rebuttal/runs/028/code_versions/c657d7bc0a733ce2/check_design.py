"""Check phase coverage, local repair geometry, holdout separation and selection."""
import hashlib
import json
from pathlib import Path

import torch

from selection import guard, select

RUN = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text())


def main():
    design = read(RUN / 'configs/design.json')
    pool = {p: read(RUN / f'configs/policies/{p}.json')['method_map'] for p in design['policies']}
    assert len(pool) == 24
    for pre in ['sq_int8', 'int8', 'dense_nvfp4']:
        for dec in ['sq_int8', 'int8', 'w4a16_ours']:
            assert any(all(v == {'prefill_method': pre, 'decode_method': dec} for v in m.values()) for m in pool.values())
    for base in ['sq_int8', 'int8']:
        previous = set()
        ranking = read(RUN / 'configs/rankings.json')[base]
        assert all(r['saved_ms'] > 0 for r in ranking)
        for count in [8, 16, 32, 64, 96, 128]:
            mapping = pool[f'repair_{base}_w4_{count:03d}']
            assert all(v['prefill_method'] == base for v in mapping.values())
            changed = {n for n, v in mapping.items() if v['decode_method'] == 'w4a16_ours'}
            assert len(changed) == count and previous <= changed
            assert changed == {r['name'] for r in ranking[:count]}
            previous = changed
    starts = sorted(design['all_new_starts'])
    assert len(starts) == 208 and all(b >= a + 2112 for a, b in zip(starts, starts[1:]))
    assert all(a + 2112 <= b or b + 2112 <= a for a in starts for b in design['excluded_calibration_starts'])
    original = torch.load(RUN.parent / '001/data/calibration_128_2112.pt', weights_only=True)
    content = lambda row: hashlib.sha256(row.numpy().tobytes()).hexdigest()
    old = {content(row) for row in original}
    seen = set()
    for split in ['validation', 'confirmation']:
        spec = design['datasets'][split]
        path = Path(spec['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
        data = torch.load(path, weights_only=True)
        assert tuple(data.shape) == (spec['blocks'], 2112)
        current = {content(row) for row in data}
        assert len(current) == len(data) and not current & (old | seen)
        seen.update(current)
    assert guard([0.] * 128, baseline=True)['guarded_delta_nll'] == 0.
    assert guard([0.] * 128)['guarded_delta_nll'] == .005
    candidates = [{'policy': 'uniform_dense_bf16', 'guarded_delta_nll': 0., 'e2e_ms': 2000., 'cycle_ms': 2000.},
        {'policy': 'uniform_int8', 'guarded_delta_nll': .009, 'e2e_ms': 1200., 'cycle_ms': 1200.},
        {'policy': 'fast_infeasible', 'guarded_delta_nll': .02, 'e2e_ms': 700., 'cycle_ms': 750.},
        {'policy': 'repair', 'guarded_delta_nll': .0095, 'e2e_ms': 1100., 'cycle_ms': 1250.}]
    assert select(candidates, .01, 'e2e_ms')[0]['policy'] == 'repair'
    assert select(candidates, .01, 'cycle_ms')[0]['policy'] == 'uniform_int8'
    assert select(candidates, .04, 'e2e_ms')[0]['policy'] == 'fast_infeasible'
    assert select(candidates, .005, 'e2e_ms')[0]['policy'] == 'uniform_dense_bf16'
    result = {'verified': True, 'policies': 24, 'new_nonoverlapping_windows': 208,
              'synthetic_checks': 'guard retained; invalid fast point excluded; E2E/cycle differ; feasible INT8 not lost'}
    (RUN / 'results/design_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()
