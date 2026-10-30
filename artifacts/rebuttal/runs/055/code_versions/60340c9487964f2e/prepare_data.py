"""Freeze model-tokenized WikiText train/validation/test inputs for run055."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import datasets
import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    data = args.run / 'data'
    data.mkdir(parents=True, exist_ok=True)
    assert not (data / 'provenance.json').exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = datasets.load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1',
                                    cache_dir=str(args.run / 'cache/datasets'))
    records = {'tokenizer': str(args.model.resolve()), 'datasets_version': datasets.__version__}
    for split in ('train', 'validation', 'test'):
        ids = tokenizer('\n\n'.join(dataset[split]['text']), add_special_tokens=False,
                        return_tensors='pt')['input_ids'][0]
        torch.save(ids, data / f'wikitext_{split}_tokens.pt')
        records[split] = {'rows': len(dataset[split]), 'tokens': len(ids),
                          'dataset_fingerprint': dataset[split]._fingerprint,
                          'cache_files': dataset[split].cache_files,
                          'token_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
        count, seed = (128, 0) if split == 'train' else (100, 1 if split == 'validation' else 42)
        rng = random.Random(seed)
        starts = [rng.randrange(len(ids) - 2112) for _ in range(count)]
        samples = torch.stack([ids[start:start + 2112] for start in starts])
        filename = {'train': 'calibration_128_2112.pt', 'validation': 'validation_100_2112.pt',
                    'test': 'evaluation_test_100_2112.pt'}[split]
        torch.save(samples, data / filename)
        records[split].update(seed=seed, starts=starts, sample_file=filename,
                              sample_shape=list(samples.shape),
                              sample_tensor_sha256=hashlib.sha256(samples.numpy().tobytes()).hexdigest())
        print(split, len(ids), records[split]['token_sha256'], flush=True)
    records['protocol'] = 'Join all raw WikiText rows by two newlines, no special tokens; independent split sampling.'
    (data / 'provenance.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
