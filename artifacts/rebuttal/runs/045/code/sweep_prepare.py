"""Freeze model-specific full-space budget/phase-weight points before evaluation."""
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

RUN=Path(__file__).resolve().parents[1]
REPO=RUN.parents[3]
sys.path.insert(0,str(REPO/'daso/src'))
from daso.modeling.phase_quality import combine_phase_quality
from exact_solver import solve

read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert not (RUN/'configs/sweep_design.json').exists()
    spec=read(RUN/'configs/model.json');source=RUN.parent/spec['source_run']
    actions=read(RUN/'reference/actions.json');q=np.array(actions['quality'])
    names,methods=actions['names'],actions['methods'];n=len(names)
    assert n==128 and q.shape==(2*n,8) and np.all(q[:,0]==0)
    profile={(r['module_type'],r['method'],int(r['m'])):float(r['latency_ms'])
             for r in csv.DictReader((RUN/'reference/profile.csv').open())}
    pairs=list(itertools.product(range(8),repeat=2))
    costs=np.array([[profile[name.rsplit('.',1)[-1],methods[p],16384]+
                     63*profile[name.rsplit('.',1)[-1],methods[d],8] for p,d in pairs]for name in names])
    raw=combine_phase_quality(q[:n],q[n:]).numpy().reshape(n,64)
    template=read(source/'configs/policies/uniform_dense_bf16.json')
    registry,signatures,points={},{},[]

    def add(pid,mapping,role):
        signature=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
        if signature in signatures:return signatures[signature]
        path=RUN/f'configs/policies/{pid}.json'
        write(path,{**template,'policy_id':pid,'method_map':mapping})
        registry[pid]={'signature':signature,'policy_sha256':sha(path),'role':role}
        signatures[signature]=pid
        return pid

    anchors=[]
    for pid in ['uniform_dense_bf16','uniform_int8','uniform_sq_int8','uniform_fp8','phase_sq_int8_w4a16_ours']:
        anchors.append(add(pid,read(source/f'configs/policies/{pid}.json')['method_map'],'reference'))
    settings=[(1.,b)for b in [.01,.02,.03,.04]]+[(w,.03)for w in [4.,32.]]
    for weight,budget in settings:
        weighted=combine_phase_quality(q[:n],q[n:],prefill_weight=weight).numpy().reshape(n,64)
        first=solve(weighted,costs,budget)
        second=solve(weighted,costs,first['latency_ms']+1e-8,inverse=True)
        assert first['optimal'] and second['optimal'] and second['quality']<=budget+1e-9
        mapping={name:{'prefill_method':methods[pairs[j][0]],'decode_method':methods[pairs[j][1]]}
                 for name,j in zip(names,second['indices'],strict=True)}
        label=f'w{weight:g}_q{budget:g}';pid=add(label,mapping,'prespecified full-space point')
        points.append({'label':label,'policy':pid,'prefill_weight':weight,'budget':budget,
            'weighted_prediction':second['quality'],'anchored_prediction':float(raw[np.arange(n),second['indices']].sum()),
            'old_bias':actions['bias'],'linear_ms':second['latency_ms'],'latency_solver':first,'quality_tie_solver':second})
        from collections import Counter
        print(label,pid,{p:dict(Counter(v[p+'_method']for v in mapping.values()))for p in ['prefill','decode']},flush=True)
    write(RUN/'configs/policy_registry.json',registry)
    write(RUN/'configs/sweep_points.json',points)
    write(RUN/'configs/selected_policies.json',{'selected':[{'policy':p}for p in registry],
        'frozen_at_unix':time.time(),'rule':'All fixed points evaluated, no task-score or NLL filtering.'})
    # Same frozen test windows as the original Llama run; explicitly not new holdout.
    original=read(source/'configs/frozen_design.json')
    dataset=original['datasets']['confirmation'].copy()
    dataset.update(file=str((source/'data/confirmation_64_2112.pt').resolve()),
        scope='Previously evaluated fixed64 WikiText test windows; not new blind confirmation.')
    assert dataset['blocks']==64 and sha(Path(dataset['file']))==dataset['sha256']
    write(RUN/'configs/frozen_design.json',{'datasets':{'confirmation':dataset},'model':spec})
    old_maps={}
    for path in sorted((source/'configs/policies').glob('*.json')):
        mapping=read(path)['method_map']
        signature=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
        old_maps.setdefault(signature,[]).append(path.stem)
    evaluation,generation,reused_nll,reused_generation=[],[],[],[]
    for pid,entry in registry.items():
        matches=old_maps.get(entry['signature'],[])
        path=next((source/f'results/confirmation/{p}/prefill_decode.json'for p in matches
                   if(source/f'results/confirmation/{p}/prefill_decode.json').exists()),None)
        if path:
            value=read(path)
            assert len(value['blocks'])==64 and value['data_sha256']==dataset['sha256']
            target=RUN/f'results/confirmation/{pid}/prefill_decode.json';target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(path.resolve())
            reused_nll.append({'policy':pid,'source':str(path.resolve()),'sha256':sha(path)})
        else:evaluation.append({'policy':pid})
        for task,count in [('cnn_dm_1000',1000),('dsum',1500),('IWSLT',333)]:
            path=next((source/f'results/generation/{p}/{task}/metrics_0000_end.json'for p in matches
                       if(source/f'results/generation/{p}/{task}/metrics_0000_end.json').exists()),None)
            if path:
                value=read(path)
                assert value['samples']==count and value['input_sha256']==sha(RUN/f'data/generation_{task}.jsonl')
                reused_generation.append({'policy':pid,'dataset':task,'source':str(path.resolve()),'sha256':sha(path)})
            else:generation.append({'policy':pid,'dataset':task})
    write(RUN/'configs/sweep_evaluation_jobs.json',{'jobs':evaluation,'reused':reused_nll})
    write(RUN/'configs/sweep_generation_jobs.json',{'jobs':generation,'reused':reused_generation})
    recipe=['generate_tasks.py','common.py','eightbit.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py','admission_check.py']
    assert all(sha(RUN/'code'/f)==sha(source/'code'/f)for f in recipe)
    inputs=[RUN/'reference/actions.json',RUN/'reference/profile.csv',REPO/'daso/src/daso/modeling/phase_quality.py',
        source/'results/final_audit.json',RUN/'configs/model.json',RUN/'configs/generation_protocol.json',
        RUN/'configs/speed_inputs.json',RUN/'data/activation_scales.pt',Path(dataset['file']),
        *[RUN/f'data/generation_{t}.jsonl'for t in ['cnn_dm_1000','dsum','IWSLT']],
        *[source/'code'/f for f in recipe]]
    design={'model':spec,'source_run':spec['source_run'],'modules':n,'actions_per_module':64,
        'settings':settings,'anchors':anchors,'unique_maps':len(registry),'dataset':dataset,
        'created_at_unix':time.time(),'input_sha256':{str(p.resolve()):sha(p)for p in inputs},
        'configuration_sha256':{p.name:sha(p)for p in (RUN/'configs').glob('*.json')},
        'new_evaluation_jobs':len(evaluation),'new_generation_jobs':len(generation),
        'scope':'Full-space exploratory P+D sensitivity; BF16-zero anchor, frozen before own GPU outputs; all results retained.'}
    write(RUN/'configs/sweep_design.json',design)
    write(RUN/'results/sweep_design_checks.json',{'verified':True,'settings':6,'unique_maps':len(registry),
        'new_nll_jobs':len(evaluation),'reused_nll_jobs':len(reused_nll),
        'new_generation_jobs':len(generation),'reused_generation_jobs':len(reused_generation)})
    print('Frozen',design['model']['model_id'],len(registry),'maps',len(evaluation),'NLL jobs',len(generation),'generation jobs',flush=True)


if __name__=='__main__':main()
