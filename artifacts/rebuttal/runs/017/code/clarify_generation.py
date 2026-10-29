"""Freeze default Llama BOS handling before any new GPU quality evaluation."""
import hashlib
import json
import shutil

from common import MODEL, RUN, write_json
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (RUN / 'results/features_complete.json').exists()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    plain = tokenizer.encode('Hello world', add_special_tokens=False)
    special = tokenizer.encode('Hello world', add_special_tokens=True)
    assert special == [tokenizer.bos_token_id, *plain], (plain, special)
    metadata = {'prompt': 'Original PMPD Claude text wrapper', 'add_special_tokens': True,
        'bos_token_id': tokenizer.bos_token_id, 'max_input_tokens': 3840,
        'truncation': 'Keep default BOS and last3839 content tokens when over length',
        'max_new_tokens':256, 'greedy':True, 'allow_eos':True, 'datasets':{}}
    archive = RUN / 'reference/generation_before_bos_clarification'
    archive.mkdir()
    for task in ['cnn_dm_1000','dsum','IWSLT']:
        path = RUN / f'data/generation_{task}.jsonl'
        original = archive / path.name
        path.rename(original)
        rows = [json.loads(line) for line in original.read_text().splitlines()]
        with path.open('w') as handle:
            for row in rows:
                ids = row['prompt_token_ids']
                assert ids[0] != tokenizer.bos_token_id
                row['prompt_token_ids'] = [tokenizer.bos_token_id, *ids[-3839:]]
                handle.write(json.dumps(row)+'\n')
        metadata['datasets'][task] = {'samples':len(rows),'sha256':sha(path),
            'before_sha256':sha(original),'reference_values_unchanged':True}
    write_json(RUN / 'configs/generation_protocol.json', metadata)
    frozen = RUN / 'configs/frozen_design.json'
    shutil.copy2(frozen, RUN / 'reference/design_before_generation_bos.json')
    design = json.loads(frozen.read_text())
    design['generation_protocol_sha256'] = sha(RUN / 'configs/generation_protocol.json')
    design['generation_clarification_scope'] = 'Before GPU work: original HF/PMPD encoding defaults add BOS for Llama; all methods use the same frozen token IDs and fresh full-task evaluation. Earlier CPU-only token files retained.'
    write_json(frozen, design)
    receipt = json.loads((RUN / 'results/cpu_preparation_complete.json').read_text())
    receipt['design_sha256'] = sha(frozen)
    write_json(RUN / 'results/cpu_preparation_complete.json', receipt)
    print('Frozen Llama BOS protocol, all earlier token files retained', flush=True)


if __name__ == '__main__':
    main()
