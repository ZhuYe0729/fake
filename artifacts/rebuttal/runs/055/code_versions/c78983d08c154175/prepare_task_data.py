"""Freeze public likelihood datasets and the original Qwen generation inputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import datasets

RUN = Path(__file__).resolve().parents[1]
BASE = RUN.parent


def freeze(item):
    task, config = item
    target = RUN / 'cache/likelihood' / task
    record = RUN / f'configs/likelihood_inputs/{task}.json'
    if record.exists():
        return json.loads(record.read_text())
    data = datasets.load_dataset(config['dataset_path'], config['dataset_name'],
                                 cache_dir=str(RUN / 'cache/datasets'))
    wanted = {config.get(key) for key in ['training_split', 'validation_split', 'test_split', 'fewshot_split']}
    wanted.discard(None)
    data = datasets.DatasetDict({name: data[name] for name in sorted(wanted)})
    target.parent.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(target))
    splits = {}
    for name, table in data.items():
        digest = hashlib.sha256()
        for row in table:
            digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())
        splits[name] = {'rows': len(table), 'logical_sha256': digest.hexdigest(),
                        'dataset_fingerprint': table._fingerprint}
    value = {'task': task, 'dataset_path': config['dataset_path'], 'dataset_name': config['dataset_name'],
             'path': str(target.resolve()), 'splits': splits}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(value, indent=2) + '\n')
    print('frozen', task, {name: row['rows'] for name, row in splits.items()}, flush=True)
    return value


def main():
    configs = json.loads((BASE / '001/data/likelihood_task_configs.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(freeze, configs.items()))
    (RUN / 'configs/likelihood_datasets.json').write_text(json.dumps(records, indent=2) + '\n')
    protocol = json.loads((BASE / '001/data/generation_protocol.json').read_text())
    for task, spec in protocol['tasks'].items():
        source = BASE / f'001/data/generation_{task}.jsonl'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['sha256']
        assert len(source.read_text().splitlines()) == spec['count']
        target = RUN / 'data' / source.name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, target)
    (RUN / 'configs/generation_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')


if __name__ == '__main__':
    main()
