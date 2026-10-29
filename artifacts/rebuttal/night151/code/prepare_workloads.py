"""Freeze run019 fixed-map workload transfer without reading new-load outcomes."""
import hashlib
import json
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / 'artifacts/rebuttal/runs'
RUN = RUNS / '019'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    assert not (RUN / 'configs/design.json').exists()
    assert read(RUNS / '016/results/final_audit.json')['verified']
    for directory in ['code', 'data', 'configs/policies', 'reference', 'logs', 'results']:
        (RUN / directory).mkdir(parents=True, exist_ok=True)
    for name in ['common.py', 'arena_runtime.py', 'eightbit.py', 'export_model.py', 'graph_bindings.py',
                 'int4_marlin.py', 'runtime_extensions.py', 'telemetry.py', 'launch.py']:
        shutil.copy2(RUNS / '016/code' / name, RUN / 'code' / name)
    for directory in ['csrc', 'compat']:
        shutil.copytree(RUNS / '016/code' / directory, RUN / 'code' / directory)
    (RUN / 'extensions').symlink_to((RUNS / '016/extensions').resolve(), target_is_directory=True)
    (RUN / 'data/activation_scales.pt').symlink_to((RUNS / '016/data/activation_scales.pt').resolve())
    launch = RUN / 'code/launch.py'
    source = launch.read_text()
    old = "TEMP = Path('/dev/shm/daso_rebuttal_016_20261026')"
    assert source.count(old) == 1
    launch.write_text(source.replace(old, "TEMP = Path('/dev/shm/daso_rebuttal_019_20261026')"))
    oldrows = read(RUNS / '015/configs/design.json')['rows']
    old = {r['method']: r['prefill_decode']['policy'] for r in oldrows}
    selections = read(RUNS / '016/configs/selected_policies.json')['selected']
    rows = [('BF16','016','uniform_dense_bf16'), ('SmoothQuant INT8','015','uniform_sq_int8'),
        ('FP8 W8A8','015','uniform_fp8'), ('W4A16 NVFP4','015','uniform_w4a16_ours'),
        ('HAWQ-FP4 r=.80','016','hawq_signed_prefill_decode_phase_pair_r0.8'),
        ('HAWQ-FP4 hardware minimum','016','hardware_minimum_prefill_decode_phase_pair'),
        ('DASO-7 exact/validation epsilon=.01','015',old['DASO + 8bit + exact/validation, epsilon=.01']),
        ('DASO-7 exact/validation epsilon=.04','015',old['DASO + 8bit + exact/validation, epsilon=.04'])]
    for epsilon in [.01,.04]:
        entry = next(x for x in selections if x['scenario']=='prefill_decode' and x['quality_budget']==epsilon)
        rows.append((f'DASO-8 exact/validation epsilon={epsilon:g}','016',entry['policy']))
    phase = read(RUNS / '016/configs/policies/uniform_dense_bf16.json')
    phase['method_map'] = {k:{'prefill_method':'sq_int8','decode_method':'w4a16_ours'} for k in phase['method_map']}
    phase['policy_id'] = 'fixed_sq_int8_w4a16'
    write(RUN / 'configs/policies/fixed_sq_int8_w4a16.json', phase)
    catalog, labels, signatures = {}, [], {}
    for label, origin, pid in rows + [('Uniform SQ-INT8 prefill / W4A16 decode','019','fixed_sq_int8_w4a16')]:
        source = RUNS / origin / f'configs/policies/{pid}.json'
        policy = read(source)
        signature = hashlib.sha256(json.dumps(policy['method_map'],sort_keys=True).encode()).hexdigest()
        key = signatures.setdefault(signature, pid)
        if key not in catalog:
            if origin != '019':shutil.copy2(source, RUN / f'configs/policies/{key}.json')
            catalog[key] = {'origin':origin,'source_policy':pid,'source_sha256':sha(source),'map_sha256':signature}
        labels.append({'method':label,'policy':key,'source_run':origin,'source_policy':pid})
    cases = [('center',8,2048,64),('single',1,2048,128),('short',8,512,128),
             ('long',8,4096,64),('decode256',8,512,256),('single256',1,512,256)]
    import torch
    from transformers import AutoTokenizer
    # Use the same local official WikiText test corpus; no new selection is done.
    import pyarrow.ipc as ipc
    cache = Path('/root/wja/.cache/huggingface/datasets/Salesforce___wikitext')
    arrows = sorted(cache.rglob('wikitext-test.arrow'))
    assert arrows, str(cache)
    with arrows[0].open('rb') as handle:
        texts = ipc.open_stream(handle).read_all().column('text').to_pylist()
    tokenizer = AutoTokenizer.from_pretrained('/root/wja/data/models/Qwen/Qwen3-8B',trust_remote_code=True)
    tokens = tokenizer('\n\n'.join(texts),add_special_tokens=False)['input_ids']
    stride = 4352
    starts = list(range(0,len(tokens)-stride+1,stride))
    assert len(starts)>=64
    rng = random.Random(151019)
    rng.shuffle(starts)
    starts = starts[:64]
    workloads = []
    for label,batch,prompt,output in cases:
        speed_input = RUN / f'configs/{label}_speed_inputs.json'
        if label in ['single','short','long']:
            original = next(w for w in read(RUNS/'015/configs/representative_workloads.json')['workloads']
                            if (w['batch'],w['prompt'],w['output'])==(batch,prompt,output))
            shutil.copy2(original['input'],speed_input)
        elif label == 'center':shutil.copy2(RUNS/'016/configs/speed_inputs.json',speed_input)
        else:
            generator = torch.Generator().manual_seed(151019+batch)
            write(speed_input,torch.randint(256,150000,(batch,prompt),generator=generator).tolist())
        tensor = torch.tensor([tokens[s:s+prompt+output] for s in starts],dtype=torch.long)
        data = RUN / f'data/{label}_quality.pt'
        torch.save(tensor,data)
        workloads.append({'label':label,'batch':batch,'prompt':prompt,'output':output,
            'input':str(speed_input),'input_sha256':sha(speed_input),'quality_file':str(data),'quality_sha256':sha(data)})
    order = sorted(catalog)
    rounds = []
    for campaign in range(3):
        current = [x for x in order if x!='uniform_dense_bf16']
        random.Random(151019+campaign).shuffle(current)
        rounds.append(['uniform_dense_bf16',*current,'uniform_dense_bf16_bracket_end'])
    write(RUN / 'configs/design.json',{'rows':labels,'policies':catalog,'workloads':workloads,
        'rounds':rounds,'warmups':3,'iterations':5,'quality_windows':64,'kv_gib':6,
        'quality_source':str(arrows[0]),'quality_source_sha256':sha(arrows[0]),'quality_starts':starts,
        'scope':'Fixed center-load maps transferred to six predeclared B/S/O loads; no per-load retuning or selection. '
                'Same local public WikiText test corpus previously benchmarked; reshuffled nonoverlapping 4352-token source blocks reused across cases, not a new unseen corpus. '
                'Three speed campaigns, BF16 start/end anchors, per-case raw TTFT/TPOT/E2E/restore/cycle. '
                'All baselines and previous DASO incumbents preserved; no method is dropped after seeing new results.'})
    write(RUN/'reference/protected_manifests.json',{r:sha(RUNS/r/'results/manifest.json') for r in ['015','016']})
    print('Frozen run019',len(labels),'labels',len(catalog),'unique maps',len(workloads),'workloads',flush=True)


if __name__ == '__main__':
    main()
