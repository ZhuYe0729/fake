from __future__ import annotations

import hashlib
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from common import MODEL, RUN, write_json


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cache = Path('/root/wja/.cache/huggingface/datasets/Salesforce___wikitext')
    files = {}
    for split in ['train', 'validation', 'test']:
        source, = list(cache.rglob(f'wikitext-{split}.arrow'))
        dataset = Dataset.from_file(str(source))
        ids = tokenizer('\n\n'.join(dataset['text']), add_special_tokens=False,
                        return_tensors='pt')['input_ids'][0]
        torch.save(ids, RUN / f'data/wikitext_{split}_tokens.pt')
        files[split] = {'source': str(source), 'tokens': len(ids),
                        'token_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
        if split == 'train':
            rng = random.Random(0)
            starts = [rng.randrange(len(ids) - 2112) for _ in range(128)]
            samples = torch.stack([ids[start:start + 2112] for start in starts])
            torch.save(samples, RUN / 'data/calibration_128_2112.pt')
            files['calibration'] = {'split': 'train', 'seed': 0, 'starts': starts,
                                    'shape': list(samples.shape)}
    write_json(RUN / 'data/provenance.json', files)
    print(files, flush=True)


if __name__ == '__main__':
    main()
