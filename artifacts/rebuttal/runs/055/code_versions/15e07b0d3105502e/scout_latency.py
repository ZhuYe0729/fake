"""Select points using validation quality and a separate exploratory timing run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from common import RUN, write_json
import formal_speed as speed
import policy_runtime
from policy_runtime import digest
from select_points import latency_shortlist, validation_candidates


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    target = RUN / 'results/validation_latency_scout.json'
    assert not target.exists()
    pool_path, pool, candidates = validation_candidates()
    shortlist = latency_shortlist(candidates, pool)
    rules_path = RUN / 'configs/selection_rules.json'
    assert pool['selection_rules_sha256'] == digest(rules_path)
    write_json(RUN / 'configs/latency_scout_shortlist.json', {'points': shortlist,
        'pool_sha256': digest(pool_path), 'rules_sha256': digest(rules_path),
        'scope': 'Frozen from validation quality and additive profile before scouting actual E2E.'})
    baseline = json.loads((RUN / 'configs/nvml_baseline.json').read_text())
    speed.NORMAL_GAP = baseline['gpu0_single_process_gap_bytes']
    speed.IDLE_OFFSETS = {row['index']: row['idle_allocated_bytes'] for row in baseline['gpus']}
    policy_runtime.invoke = speed.run_owned
    label = f'validation_scout_{time.time_ns()}'
    records = []
    with (RUN / 'gpu0.lock').open('a') as lock0, (RUN / 'gpu1.lock').open('a') as lock1:
        fcntl.flock(lock0, fcntl.LOCK_EX); fcntl.flock(lock1, fcntl.LOCK_EX)
        speed.wait_idle(label)
        records.append(speed.measure('uniform_dense_bf16', None, label + '_bf16_start', 0))
        for policy in dict.fromkeys(row['policy_id'] for row in shortlist if row['policy_id'] != 'uniform_dense_bf16'):
            scenarios = {row['scenario'] for row in shortlist if row['policy_id'] == policy}
            scenario = None if len(scenarios) == 2 else next(iter(scenarios))
            records.append(speed.measure(policy, scenario, label + '_' + policy, 0))
        records.append(speed.measure('uniform_dense_bf16', None, label + '_bf16_end', 0))
    points = {}
    for source in records:
        body = json.loads(Path(source['path']).read_text())
        mapping = json.loads((RUN / f"configs/policies/{source['policy']}.json").read_text())['method_map']
        signature = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        for case in body['cases']:
            key = case['case']['scenario'] + '_' + signature
            if key in points:
                assert source['policy'] == 'uniform_dense_bf16'
                points[key]['medians'] = {metric: (value + case['medians'][metric]) / 2
                    for metric, value in points[key]['medians'].items()}
                points[key]['bracket_end'] = source
            else:
                points[key] = {'policy_id': source['policy'], 'source': source, 'medians': case['medians']}
    assert all(row['scenario'] + '_' + row['map_signature'] in points for row in shortlist)
    write_json(target, {'complete': True, 'points': points, 'records': records,
        'pool_sha256': digest(pool_path), 'selection_rules_sha256': digest(rules_path),
        'scope': 'Validation-time latency exploration only. Same GPU0/CPU0-7/NUMA0 and both locks; '
                 'five warmups and ten timed samples. Formal three-round timings are separate and follow policy freeze.'})


if __name__ == '__main__':
    main()
