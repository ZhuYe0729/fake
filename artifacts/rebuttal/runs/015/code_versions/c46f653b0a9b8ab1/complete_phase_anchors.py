"""Cover every prefill action with training/profile-nondominated decode anchors."""
import hashlib
import json
import shutil
import time

import numpy as np

from common import RUN, write_json


def main():
    reference = RUN / 'reference/initial_reanalysis_pool'
    assert not reference.exists()
    reference.mkdir(parents=True)
    for name in ['reanalysis_pool.json', 'selection_rules.json', 'validation_jobs.json']:
        shutil.copy2(RUN / 'configs' / name, reference / name)
    pool = json.loads((RUN / 'configs/reanalysis_pool.json').read_text())
    actions = json.loads((RUN / 'data/graph_actions_prefill_decode.json').read_text())
    methods, names = actions['methods'], actions['names']
    q = np.array(actions['quality'])[len(names):].sum(0)
    t = np.array(actions['latency_ms'])[len(names):].sum(0)
    frontier = [i for i in range(len(methods)) if not any(q[j] <= q[i] and t[j] <= t[i]
        and (q[j] < q[i] or t[j] < t[i]) for j in range(len(methods)) if j != i)]
    signatures = {v['signature']: pid for pid, v in pool['policies'].items()}
    template = json.loads((RUN / 'configs/policies/uniform_dense_bf16.json').read_text())
    jobs = []
    added = []
    for pre in methods:
        for i in frontier:
            dec = methods[i]
            mapping = {n: {'prefill_method': pre, 'decode_method': dec} for n in names}
            signature = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
            pid = signatures.get(signature)
            if pid is None:
                pid = 'graph_phase_' + pre + '_' + dec
                write_json(RUN / f'configs/policies/{pid}.json', {**template, 'policy_id': pid,
                    'scenario': 'prefill_decode', 'method_map': mapping})
                pool['policies'][pid] = {'signature': signature, 'scenarios': ['prefill_decode'], 'daso_eligible': True}
                signatures[signature] = pid
                jobs.append({'policy': pid, 'scenarios': ['prefill_decode']})
                added.append(pid)
            elif 'prefill_decode' not in pool['policies'][pid]['scenarios']:
                pool['policies'][pid]['scenarios'].append('prefill_decode')
                jobs.append({'policy': pid, 'scenarios': ['prefill_decode']})
            pool['points'].append({'label': 'anchor_' + pre + '_' + dec, 'policy_id': pid,
                'scenario': 'prefill_decode', 'family': 'systematic phase-uniform anchor'})
    amendment = {'time_ns': time.time_ns(), 'added_maps': added,
        'decode_frontier': [methods[i] for i in frontier],
        'decoder_training_quality_costs': q.tolist(), 'decoder_measured_costs_ms': t.tolist(),
        'reason': 'Complete phase-uniform coverage on the training-surrogate/hardware decode frontier, including high-quality BF16 decode anchors.',
        'data_use': 'Uses only the frozen training quality surrogate and new hardware profile; no new validation/test values were read to add anchors. Initial validation was already running; original pool/rules/job list retained.'}
    pool['anchor_amendment'] = amendment
    write_json(RUN / 'configs/reanalysis_pool.json', pool)
    rules = json.loads((RUN / 'configs/selection_rules.json').read_text())
    rules['pool_sha256'] = hashlib.sha256((RUN / 'configs/reanalysis_pool.json').read_bytes()).hexdigest()
    rules['anchor_amendment'] = amendment
    write_json(RUN / 'configs/selection_rules.json', rules)
    write_json(RUN / 'configs/validation_additional_jobs.json', {'jobs': jobs, 'amendment': amendment})
    print('Additional maps', len(added), 'additional quality jobs', len(jobs), 'frontier', amendment['decode_frontier'])


if __name__ == '__main__':
    main()
