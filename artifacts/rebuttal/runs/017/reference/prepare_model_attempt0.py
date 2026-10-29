"""Freeze model-specific data, old-label provenance, and eight-action recipes."""
import csv
import hashlib
import importlib.util
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from common import MODEL, REPO, RUN, SPEC, write_json

METHODS = ['dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4',
           'w4a16_ours', 'sq_int8', 'fp8', 'int8']
KINDS = ['self_attn.qkv_proj', 'self_attn.o_proj', 'mlp.gate_up_proj', 'mlp.down_proj']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (RUN / 'configs/frozen_design.json').exists()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    names = [f'model.layers.{i}.{kind}' for i in range(SPEC['layers']) for kind in KINDS]
    template = {'default_prefill_method': 'dense_bf16', 'default_decode_method': 'dense_bf16',
                'modules_to_not_convert': ['lm_head']}
    for method in METHODS + ['sq_fp8', 'marlin_int4']:
        write_json(RUN / f'configs/policies/uniform_{method}.json', {
            **template, 'policy_id': 'uniform_' + method, 'scenario': 'prefill_decode',
            'method_map': {n: {'prefill_method': method, 'decode_method': method} for n in names}})
    datasets, provenance, token_splits = {}, {}, {}
    for split in ['train', 'validation', 'test']:
        sources = list((REPO / 'artifacts/rebuttal/runs/001/data/likelihood_datasets').rglob(f'wikitext-{split}.arrow'))
        assert len(sources) == 1, sources
        source = sources[0]
        raw = Dataset.from_file(str(source))
        ids = tokenizer('\n\n'.join(raw['text']), add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        torch.save(ids, RUN / f'data/wikitext_{split}_tokens.pt')
        token_splits[split] = ids
        provenance[split] = {'source': str(source), 'source_sha256': sha(source), 'tokens': len(ids),
                             'token_file_sha256': sha(RUN / f'data/wikitext_{split}_tokens.pt')}
        if split != 'train':
            count = 32 if split == 'validation' else 64
            label = 'validation' if split == 'validation' else 'confirmation'
            starts = list(range(0, len(ids) - 2112 + 1, 2112))
            seed = 15100000 + int(SPEC['run'])*10 + (1 if split == 'validation' else 3)
            random.Random(seed).shuffle(starts)
            starts = starts[:count]
            assert len(starts) == count
            target = RUN / f'data/{label}_{count}_2112.pt'
            torch.save(torch.stack([ids[s:s+2112] for s in starts]), target)
            datasets[label] = {'file': str(target), 'blocks': count, 'starts': starts,
                'excluded_old_starts': [], 'source': str(RUN / f'data/wikitext_{split}_tokens.pt'),
                'sha256': sha(target), 'seed': seed,
                'scope': 'Fresh frozen windows for this rebuttal run; benchmark has prior aggregate evaluation, not a wholly unseen corpus.'}
    old_pp = Path(SPEC['prefill_original'])
    pp_files = list((old_pp / 'samples').glob('*.pt'))
    assert len(pp_files) == 1
    prefill = torch.load(pp_files[0], weights_only=True)
    SPEC['training_prefill_length'] = int(prefill.shape[1])
    assert prefill.shape[0] >= 100 and SPEC['training_prefill_length'] in [2048, 2049]
    shutil.copy2(pp_files[0], RUN / 'data/calibration_train_100_2048.pt')
    pd_file = Path(SPEC['decode_original']) / 'samples/wikitext_2048_64.pt'
    decode = torch.load(pd_file, weights_only=True)[:, :2112]
    assert decode.shape[0] >= 100
    if len(decode) < 128:
        rng = random.Random(151001 + int(SPEC['run']))
        ids = token_splits['train']
        extra = [ids[s:s+2112] for s in [rng.randrange(len(ids)-2112) for _ in range(128-len(decode))]]
        decode = torch.cat([decode, torch.stack(extra)])
    torch.save(decode[:128], RUN / 'data/calibration_128_2112.pt')
    write_json(RUN / 'configs/model.json', SPEC)
    provenance['training'] = {'prefill_source': str(pp_files[0]), 'prefill_sha256': sha(pp_files[0]),
        'decode_source': str(pd_file), 'decode_source_sha256': sha(pd_file),
        'decode_first100_unchanged': True, 'extra_contexts': 28 if len(torch.load(pd_file, weights_only=True)) == 100 else 0}
    old = Path(SPEC['old_modeling'])
    jobs = []
    for scene in ['prefill_only', 'prefill_decode']:
        manifest = []
        original_manifest = read(old / scene / 'policies/manifest.json')
        assert len(original_manifest) == 72
        raw_root = old_pp if scene == 'prefill_only' else Path(SPEC['decode_original'])
        for entry in original_manifest:
            original_id = entry['policy_id']
            pid = f'legacy_{scene}_{original_id}'
            original_policy = old / scene / f'policies/{original_id}.json'
            policy = read(original_policy)
            assert set(policy['method_map']) == set(names)
            write_json(RUN / f'configs/policies/{pid}.json', {**policy, 'policy_id': pid})
            candidates = [raw_root / f'nll/raw/{original_id}.json', raw_root / f'results/{original_id}.json']
            source, = [x for x in candidates if x.exists()]
            label = read(source)
            assert len(label['blocks']) == 100 and isinstance(label['avg_nll'], float)
            target = RUN / f'results/training/{scene}/{pid}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
            manifest.append({'policy_id': pid, 'split': entry['split'], 'origin': 'paper',
                'source_sha256': sha(source), 'original_policy_sha256': sha(original_policy),
                'policy_sha256': sha(RUN / f'configs/policies/{pid}.json')})
        rng = np.random.default_rng(151000 + int(SPEC['run'])*10 + (1 if scene == 'prefill_only' else 2))
        rates = np.geomspace(.01, 1., 45)
        rng.shuffle(rates)
        for index in range(48):
            pid = f'cal{SPEC["run"]}_{scene}_{index:02d}'
            probabilities = rng.dirichlet(np.ones(7))
            mapping = {}
            for name in names:
                if index < 3:
                    pre = dec = ['sq_int8', 'fp8', 'int8'][index]
                else:
                    pre = str(rng.choice(METHODS[1:], p=probabilities)) if rng.random() < rates[index-3] else 'dense_bf16'
                    dec = str(rng.choice(METHODS[1:], p=probabilities)) if rng.random() < rates[index-3] else 'dense_bf16'
                    if scene == 'prefill_only':
                        dec = pre
                mapping[name] = {'prefill_method': pre, 'decode_method': dec}
            write_json(RUN / f'configs/policies/{pid}.json', {**template, 'policy_id': pid,
                'scenario': scene, 'method_map': mapping})
            item = {'policy_id': pid, 'split': 'train' if index < 36 else 'holdout',
                'origin': SPEC['run'], 'policy_sha256': sha(RUN / f'configs/policies/{pid}.json')}
            manifest.append(item)
            jobs.append({'job_id': pid, 'policy': pid, 'scenarios': [scene], 'policy_sha256': item['policy_sha256']})
        write_json(RUN / f'configs/calibration_{scene}.json', manifest)
        for phase in ['prefill', 'decode']:
            for method in METHODS[1:5]:
                source = old / scene / f'local_errors/{phase}_{method}.csv'
                if not source.exists():
                    source = old / 'prefill_decode' / f'local_errors/{phase}_{method}.csv'
                destination = RUN / f'reference/local_errors/{scene}/{phase}_{method}.csv'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    write_json(RUN / 'configs/training_jobs.json', jobs)
    for filename, seed in [('speed_inputs.json', 144983), ('calibration_speed_inputs.json', 151000+int(SPEC['run']))]:
        values = torch.randint(100, int(read(MODEL / 'config.json')['vocab_size'])-1,
            (8, 2048), generator=torch.Generator().manual_seed(seed))
        write_json(RUN / 'configs' / filename, values.tolist())
    for name in ['likelihood_datasets', 'likelihood_auxiliary_datasets']:
        source = REPO / ('artifacts/rebuttal/runs/001/data/' if name == 'likelihood_datasets' else 'artifacts/rebuttal/runs/002/data/') / name
        (RUN / 'data' / name).symlink_to(source.resolve(), target_is_directory=True)
    # Freeze original paper task texts with each model's tokenizer. No network.
    source = REPO / 'references/pmpd_eval_kit/pmpd_eval.py'
    spec = importlib.util.spec_from_file_location('paper_tasks', source)
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    from types import SimpleNamespace
    for dataset, count in [('cnn_dm_1000', 1000), ('dsum', 1500), ('IWSLT', 333)]:
        questions = tasks.build_questions(SimpleNamespace(dataset=dataset, split='test',
            data_root=Path('/root/wja/data/datasets/flaxquant'),
            iwslt_filter_tokenizer=str(REPO / 'artifacts/rebuttal/runs/001/reference/vicuna_tokenizer'),
            question_begin=None, question_end=None))
        assert len(questions) == count
        with (RUN / f'data/generation_{dataset}.jsonl').open('w') as handle:
            for q in questions:
                handle.write(json.dumps({'question_id': q['question_id'], 'reference': q['reference'],
                    'prompt_token_ids': tokenizer.encode(q['prompt'], add_special_tokens=False)[-3840:]})+'\n')
    design = {'model': SPEC, 'methods': METHODS, 'names': names, 'datasets': datasets,
        'new_labels_per_scenario': 48, 'inherited_labels_per_scenario': 72,
        'quality_fit': {'steps': 4000, 'learning_rate': .03, 'regularization': .05,
                        'architecture': 'Original positive factored calibrator; original bucket error features, new methods measured on target model.'},
        'quality_budgets': [.01, .04], 'latency_fractions': [.95, .80, .65] if SPEC['run']=='017' else [.80],
        'matched_pd_modes': ['same_operator', 'phase_pair'], 'validation_guard': {'z': 2., 'margin': .005},
        'label_reuse_gate': 'BF16 and uniform NVFP4 on both original sample tensors; mean tolerance1e-4,block tolerance1e-3. If failure, recompute old labels before fitting.',
        'training_manifest_sha256': {s:sha(RUN / f'configs/calibration_{s}.json') for s in ['prefill_only','prefill_decode']},
        'training_jobs_sha256': sha(RUN / 'configs/training_jobs.json'),
        'speed_rounds': 3, 'warmups': 3, 'timed_per_round': 8,
        'scope': 'New model-specific rebuttal run, all allocations fixed before validation; selection fixed before confirmation. Historical paper results retained.'}
    write_json(RUN / 'configs/frozen_design.json', design)
    write_json(RUN / 'data/provenance.json', provenance)
    write_json(RUN / 'results/cpu_preparation_complete.json', {'complete': True, 'new_training_jobs': len(jobs),
        'names': len(names), 'training_prefill_length': SPEC['training_prefill_length'],
        'design_sha256': sha(RUN / 'configs/frozen_design.json')})
    print('Prepared', SPEC['model_id'], 'new jobs', len(jobs), flush=True)


if __name__ == '__main__':
    main()
