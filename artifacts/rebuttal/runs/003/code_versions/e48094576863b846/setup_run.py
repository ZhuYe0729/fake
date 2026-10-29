"""Freeze the workload sweep before profiling or test evaluation."""
import hashlib
import json
import shutil
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
OLD = RUN.parent / '002'


def main():
    for directory in ['configs/policies', 'data', 'reference', 'results', 'logs', 'extensions']:
        (RUN/directory).mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ['common.py', 'eightbit.py', 'runtime_extensions.py', 'export_model.py', 'int4_marlin.py']:
        source=OLD/'code'/name; target=RUN/'code'/name
        if not target.exists():shutil.copy2(source,target)
        copied[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copytree(OLD/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    for name in ['activation_scales.pt','profile_weights.pt','local_errors.csv','wikitext_test_tokens.pt']:
        target=RUN/'data'/name
        if not target.exists():target.symlink_to('../../002/data/'+name)
    for scenario in ['prefill_only','prefill_decode']:
        target=RUN/f'reference/quality_model_{scenario}.json'
        if not target.exists():shutil.copy2(OLD/f'results/daso8/{scenario}/quality_model.json',target)
    for rid, name in [('001','hessian_trace_blockwise_p16_b4_s256.json'),
                      ('001','weight_quantization_errors.csv'),('001','marlin_int4_weight_errors.csv')]:
        target=RUN/'reference'/name
        if not target.exists():shutil.copy2(RUN.parent/rid/'results'/name,target)
    cases=[(1,2048,64),(4,2048,64),(8,2048,64),(16,2048,64),(8,512,64),
           (8,4096,64),(8,2048,16),(8,2048,256),(1,512,256)]
    workloads=[]
    for b,s in dict.fromkeys((b,s) for b,s,o in cases):
        workloads.append({'id':f'p_b{b}_s{s}','scenario':'prefill_only','batch':b,'input':s,'output':1})
    for b,s,o in cases:
        workloads.append({'id':f'pd_b{b}_s{s}_o{o}','scenario':'prefill_decode','batch':b,'input':s,'output':o})
    design={'workloads':workloads,'quality_budgets':[.01,.07],
        'methods':['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours','sq_int8','fp8'],
        'uniform_methods':['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours','int8','marlin_int4','sq_int8','fp8'],
        'kv_bytes':6*1024**3,'warmups':2,'iterations':5,'quality_blocks':100,
        'speed_gpu':3,'quality_gpus':[0,4,7],
        'engine':{'max_model_len':4360,'max_num_seqs':16,'max_num_batched_tokens':32768,
                  'enforce_eager':True,'prefix_caching':False,'chunked_prefill':False},
        'scope':'synchronous homogeneous batches; no claim about mixed continuous batching',
        'quality_transfer':'frozen run002 calibration and local errors; no refit or test feedback',
        'solver':'unchanged 2000-bin conservative DP',
        'ordinary_int8':'same weight quantizer; fused token quantization and output scaling, numeric audit required',
        'selection':'all workloads, budgets and controls frozen before new measurements'}
    path=RUN/'configs/design.json'
    if path.exists():assert json.loads(path.read_text())==design
    else:path.write_text(json.dumps(design,indent=2)+'\n')
    import random
    import torch
    tokens=torch.load(RUN/'data/wikitext_test_tokens.pt',weights_only=True).reshape(-1)
    rng=random.Random(143731)
    starts=[rng.randrange(len(tokens)-4352+1) for _ in range(100)]
    samples=torch.stack([tokens[i:i+4352] for i in starts])
    target=RUN/'data/test_100_4352.pt'
    if not target.exists():torch.save(samples,target)
    else:assert torch.equal(torch.load(target,weights_only=True),samples)
    metadata={'split':'WikiText-2 test','seed':143731,'starts':starts,'shape':list(samples.shape),
        'same_blocks':'prefix slices of the same 100 blocks for every policy/workload',
        'bootstrap':'paired block bootstrap, descriptive interval; overlapping text windows are not independent documents',
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (RUN/'data/test_samples.json').write_text(json.dumps(metadata,indent=2)+'\n')
    source={'copied_run002_code_sha256':copied,'inherited_manifests':{
        rid:hashlib.sha256((RUN.parent/rid/'results/manifest.json').read_bytes()).hexdigest() for rid in ['001','002']}}
    (RUN/'reference/sources.json').write_text(json.dumps(source,indent=2)+'\n')
    print(json.dumps({'workloads':len(workloads),'sample_sha256':metadata['sha256']}))


if __name__=='__main__':main()
