"""Match existing labels by effective precision map and exact dataset hash."""
import hashlib
import json
from pathlib import Path

import torch

from common import RUN, write_json


def signature(mapping, scenario):
    effective = ({n: p['prefill_method'] for n, p in mapping.items()}
                 if scenario == 'prefill_only' else mapping)
    return hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()


def main():
    assert not (RUN / 'configs/validation_jobs.json').exists()
    for split in ['validation', 'test']:
        target = RUN / f'data/{split}_64_2112.pt'
        if not target.exists():
            target.symlink_to((RUN.parent / f'005/data/{split}_64_2112.pt').resolve())
    inputs = torch.randint(100, 30000, (8, 2048), generator=torch.Generator().manual_seed(148015771)).tolist()
    write_json(RUN / 'configs/calibration_speed_inputs.json', inputs)
    dataset_sha = hashlib.sha256((RUN / 'data/validation_64_2112.pt').read_bytes()).hexdigest()
    cache = {}
    for rid in ['005', '012']:
        previous = RUN.parent / rid
        for source in sorted((previous / 'results/validation').glob('*/*.json')):
            data = json.loads(source.read_text())
            if data.get('data_sha256') != dataset_sha or len(data.get('blocks', [])) != 64:
                continue
            policy = previous / f'configs/policies/{source.parent.name}.json'
            if not policy.exists():
                continue
            scenario = source.stem
            key = signature(json.loads(policy.read_text())['method_map'], scenario), scenario
            cache.setdefault(key, source)
    pool = json.loads((RUN / 'configs/reanalysis_pool.json').read_text())
    jobs, inherited = [], []
    for pid, spec in pool['policies'].items():
        if not spec['daso_eligible']:
            continue
        mapping = json.loads((RUN / f'configs/policies/{pid}.json').read_text())['method_map']
        pending = []
        for scenario in spec['scenarios']:
            key = signature(mapping, scenario), scenario
            source = cache.get(key)
            if source is None:
                pending.append(scenario)
                continue
            target = RUN / f'results/validation/{pid}/{scenario}.json'
            assert not target.exists()
            data = json.loads(source.read_text())
            data['inherited_source'] = str(source)
            data['effective_map_sha256'] = key[0]
            data['inheritance_scope'] = 'Same effective precision map, canonical weights, unchanged numerical operations, exact frozen dataset and batch8.'
            write_json(target, data)
            inherited.append({'policy': pid, 'scenario': scenario, 'source': str(source),
                              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        if pending:
            jobs.append({'policy': pid, 'scenarios': pending})
    write_json(RUN / 'configs/validation_jobs.json', {'jobs': jobs, 'inherited': inherited,
        'source_data_sha256': dataset_sha, 'scope': 'New labels only for eligible DASO selection candidates lacking exact prior labels.'})
    print('Validation jobs', len(jobs), 'new scenarios', sum(len(j['scenarios']) for j in jobs),
          'inherited scenarios', len(inherited), flush=True)


if __name__ == '__main__':
    main()
