from __future__ import annotations

import hashlib
import importlib.util
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from common import MODEL, REPO, RUN, write_json


def main():
    source = REPO / 'references/pmpd_eval_kit/pmpd_eval.py'
    spec = importlib.util.spec_from_file_location('paper_pmpd_eval', source)
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    filter_root = RUN / 'reference/vicuna_tokenizer'
    filter_root.mkdir(exist_ok=True)
    info = json.load(urllib.request.urlopen('https://huggingface.co/api/models/lmsys/vicuna-7b-v1.5', timeout=30))
    revision = info['sha']
    for item in info['siblings']:
        name = item['rfilename']
        if name not in ['config.json', 'tokenizer.model', 'tokenizer.json', 'tokenizer_config.json',
                        'special_tokens_map.json', 'added_tokens.json']:
            continue
        data = urllib.request.urlopen(f'https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/{revision}/{name}',timeout=60).read()
        (filter_root / name).write_bytes(data)
    # Keep the paper's task instructions and sample filtering, replacing only
    # its legacy Claude chat wrapper with the native Qwen non-thinking wrapper.
    paper.claude_prompt = lambda message: tokenizer.apply_chat_template(
        [{'role': 'user', 'content': message}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    metadata = {'prompt': 'native Qwen3 chat template; enable_thinking=False',
                'max_input_tokens': 3840, 'max_new_tokens': 256,
                'truncation': 'left truncate encoded prompt, preserving special-token IDs',
                'sample_filter': 'original PMPD Vicuna tokenizer, English reference length > 60',
                'filter_tokenizer_revision': revision,
                'paper_loader_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'tasks': {}}
    for name, count in [('cnn_dm_1000',1000), ('dsum',1500), ('IWSLT',333)]:
        args = SimpleNamespace(dataset=name, split='test', data_root=Path('/root/wja/data/datasets/flaxquant'),
            iwslt_filter_tokenizer=str(filter_root), question_begin=None, question_end=None)
        questions = paper.build_questions(args)
        if len(questions) != count:
            raise ValueError(f'{name}: expected original {count} samples, got {len(questions)}')
        target = RUN / f'data/generation_{name}.jsonl'
        with target.open('w') as handle:
            for question in questions:
                ids = tokenizer.encode(question['prompt'], add_special_tokens=False)[-3840:]
                handle.write(json.dumps({'question_id': question['question_id'],
                    'prompt_token_ids': ids, 'reference': question['reference']},ensure_ascii=False)+'\n')
        metadata['tasks'][name] = {'count':len(questions), 'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        print(name, len(questions), flush=True)
    write_json(RUN / 'data/generation_protocol.json', metadata)


if __name__ == '__main__':
    main()
