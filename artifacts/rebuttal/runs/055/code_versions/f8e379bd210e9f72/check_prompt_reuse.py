"""Measure tied-phase equivalence before sharing prompt calibration labels."""
import hashlib
import json
import time

from common import RUN, write_json
from policy_runtime import quality


def main():
    target = RUN / 'results/prompt_label_reuse_check.json'
    assert not target.exists()
    source_policy = 'cal32_p08'
    source = json.loads((RUN / f'configs/policies/{source_policy}.json').read_text())
    name = 'prompt_tied_check_p08'
    mapping = {module: {'prefill_method': pair['prefill_method'], 'decode_method': pair['prefill_method']}
               for module, pair in source['method_map'].items()}
    path = RUN / f'configs/policies/{name}.json'
    write_json(path, {**source, 'policy_id': name, 'scenario': 'prefill_only', 'method_map': mapping})
    design = json.loads((RUN / 'configs/design.json').read_text())
    design['policies'][name] = {'display': name, 'scenario': 'prefill_only',
                              'map_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(RUN / 'configs/design.json', design)
    case = next(case for case in design['workloads'] if case['scenario'] == 'prefill_only')
    original = RUN / f"results/quality/calibration/{source_policy}/{case['label']}.json"
    assert original.exists()
    measured, = quality(name, 'calibration', f'prompt_reuse_{time.time_ns()}')
    a, b = [json.loads(item.read_text()) for item in [original, measured]]
    assert a['sample_sha256'] == b['sample_sha256'] and len(a['blocks']) == len(b['blocks']) == 100
    differences = [new['avg_nll'] - old['avg_nll'] for old, new in zip(a['blocks'], b['blocks'], strict=True)]
    assert max(abs(value) for value in differences) <= 1e-5
    write_json(target, {'complete': True, 'source': str(original), 'tied': str(measured),
        'windows': 100, 'max_absolute_delta_nll': max(abs(value) for value in differences),
        'scope': 'Same effective prefill map with different decode maps. Prompt labels are reusable; no claim of new independent text.'})


if __name__ == '__main__':
    main()
