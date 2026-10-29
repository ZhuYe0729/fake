import hashlib
import json
from pathlib import Path

import torch

RUN = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    design = read(RUN / 'configs/design.json')
    pool = read(RUN / 'configs/policy_registry.json')
    assert len(pool) == 36 and sha(RUN / 'configs/policy_registry.json') == design['calibration_registry_sha256']
    assert not set(design['fit_policies']) & set(design['holdouts'])
    assert len(design['fit_policies']) == 26 and len(design['holdouts']) == 8
    for path, digest in design['source_sha256'].items():
        assert sha(Path(path)) == digest, path
    for pid, item in pool.items():
        policy = RUN / f'configs/policies/{pid}.json'
        assert sha(policy) == item['policy_sha256']
        mapping = read(policy)['method_map']
        if item['mask'] is not None:
            expected = {n for g, names in enumerate(design['groups']) if item['mask'] & (1 << g) for n in names}
            actual = {n for n, p in mapping.items() if p['decode_method'] == 'w4a16_ours'}
            assert expected == actual and all(p['prefill_method'] == 'sq_int8' for p in mapping.values())
    starts = sorted(design['all_new_starts'])
    assert len(starts) == 272 and all(b >= a + 2112 for a, b in zip(starts, starts[1:]))
    assert all(a + 2112 <= b or b + 2112 <= a for a in starts for b in design['excluded_starts'])
    previous = set()
    for split in ['validation', 'confirmation']:
        spec = design['datasets'][split]
        assert sha(Path(spec['file'])) == spec['sha256']
        data = torch.load(spec['file'], weights_only=True)
        assert data.shape == (128, 2112)
        values = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in data}
        assert len(values) == 128 and not values & previous
        previous.update(values)
    result = {'verified': True, 'calibration_maps': 36, 'fit_policies': 26,
              'heldout_policies': 8, 'new_token_disjoint_windows': 272}
    (RUN / 'results/design_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()
