"""Freeze exact/inverse candidate pool and train/validation/test protocol."""
import csv
import hashlib
import itertools
import json
import resource
import shutil
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def main():
    old=RUN.parent/'003'
    for d in ['configs/policies','data','reference','results','logs','code/compat']:(RUN/d).mkdir(parents=True,exist_ok=True)
    for n in ['common.py','eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py']:
        if not (RUN/'code'/n).exists():shutil.copy2(old/'code'/n,RUN/'code'/n)
    shutil.copytree(old/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    for name in ['activation_scales.pt','local_errors.csv']:
        if not (RUN/'data'/name).exists():(RUN/'data'/name).symlink_to(old/'data'/name)
    sys.path.insert(0,str(RUN/'code'))
    from common import write_json
    from daso.optimization import Action,ModuleChoices,solve_budget
    from exact_solver import solve
    import numpy as np
    import torch
    methods=['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours','sq_int8','fp8']
    policy=json.loads((old/'configs/policies/uniform_dense_bf16.json').read_text());names=list(policy['method_map'])
    kinds=['qkv_proj','o_proj','gate_up_proj','down_proj']
    errors={(r['module'],r['phase'],r['method']):float(r['output_rel_mse']) for r in csv.DictReader((RUN/'data/local_errors.csv').open())}
    profile={(r['module_type'],r['method'],int(r['m'])):float(r['latency_ms']) for r in csv.DictReader((old/'results/action_profile.csv').open())}
    # Independent, nonoverlapping within each split. Surrogate/canonical preparation
    # used WikiText train; validation is now used only for fixed candidate selection.
    sources={};blocks=64;length=2112
    for split,seed in [('validation',144771),('test',144773)]:
        tokens=torch.load(RUN.parent/f'001/data/wikitext_{split}_tokens.pt',weights_only=True)
        starts=np.arange(len(tokens)//length)*length
        rng=np.random.default_rng(seed);rng.shuffle(starts);starts=starts[:blocks].tolist()
        target=RUN/f'data/{split}_64_2112.pt';torch.save(torch.stack([tokens[s:s+length] for s in starts]),target)
        sources[split]={'split':split,'seed':seed,'starts':starts,'blocks':blocks,'length':length,
            'file_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'nonoverlap_within_split':True}
    rules={'quality_budgets':[.005,.01,.02,.04],'pool_budget_fractions':[0.,.25,.5,.75,1.],
        'inverse_latency_fractions':[.95,.80,.65],'quality_guard':{'z':2.,'additive_delta_nll':.005},
        'latency_guard_fraction':.05,'selection':'Among candidates with validation mean delta + 2 paired SE + .005 <= requested quality cap, choose minimum common-card measured calibration latency. BF16 reference is exact zero and exempt from added quality guard. Inverse: minimum guarded validation quality subject to 5% guarded measured calibration latency SLA.',
        'speed_calibration_seed':144881,'speed_test_seed':144883,
        'test_policy':'No test label read until configs/selected_policies.json is written and hashed. Final test evaluates preregistered original/exact points plus selected verification/inverse maps and three uniform anchors.',
        'workload':{'batch':8,'input':2048,'output':64},'kv_bytes':6*1024**3,
        'warmups':2,'iterations':5,'quality_score_batch':8,'data':sources,'methods':methods,
        'scope':'Frozen run002 quality fit and run003 measured additive Linear costs; no refit from test; MILP variant and extra validation stage distinguished from original DP.'}
    write_json(RUN/'configs/design.json',rules)
    signatures={};catalog={};points=[];scaling=[]

    def add(label,mapping,scenario,family,**meta):
        signature=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
        pid=signatures.setdefault(signature,label)
        if pid not in catalog:
            export={**policy,'policy_id':pid,'scenario':scenario,'method_map':mapping}
            write_json(RUN/f'configs/policies/{pid}.json',export);catalog[pid]={'signature':signature,'scenarios':[]}
        if scenario not in catalog[pid]['scenarios']:catalog[pid]['scenarios'].append(scenario)
        points.append({'label':label,'policy_id':pid,'scenario':scenario,'family':family,**meta})

    def mapping(indices,phases):
        output={name:{'prefill_method':'dense_bf16','decode_method':'dense_bf16'} for name in names}
        for i,phase in enumerate(phases):
            for j,name in enumerate(names):output[name][phase+'_method']=methods[indices[i*len(names)+j]]
        if len(phases)==1:
            for value in output.values():value['decode_method']=value['prefill_method']
        return output

    for scenario in ['prefill_only','prefill_decode']:
        fit=json.loads((old/f'reference/quality_model_{scenario}.json').read_text())
        phases=['prefill'] if scenario=='prefill_only' else ['prefill','decode'];q=[];t=[]
        for pi,phase in enumerate(phases):
            for name in names:
                bucket=int(name.split('.')[2])*4//36;kind=kinds.index(name.rsplit('.',1)[-1]);qr=[];tr=[]
                for mi,method in enumerate(methods):
                    qr.append(0. if method=='dense_bf16' else errors[name,phase,method]*fit['coefficients'][pi][mi][bucket][kind]/fit['feature_scale'])
                    tr.append(profile[kinds[kind],method,16384 if phase=='prefill' else 8]*(1 if phase=='prefill' else 63))
                q.append(qr);t.append(tr)
        q=np.array(q);t=np.array(t)
        write_json(RUN/f'data/actions_{scenario}.json',{'quality':q.tolist(),'latency_ms':t.tolist(),'names':names,'phases':phases,'bias':fit['bias'],'methods':methods})
        for method in ['dense_bf16','sq_int8','fp8']:
            indices=[methods.index(method)]*len(q)
            add('uniform_'+method,mapping(indices,phases),scenario,'uniform',predicted_delta_nll=fit['bias']+float(q[:,methods.index(method)].sum()),predicted_linear_ms=float(t[:,methods.index(method)].sum()))
        budgets=sorted({round(e*f,10) for e in rules['quality_budgets'] for f in rules['pool_budget_fractions']})
        for budget in budgets:
            result=solve(q,t,budget-fit['bias'])
            add(f'exact_{scenario}_q{budget:g}',mapping(result['indices'],phases),scenario,'exact forward',quality_budget=budget,
                predicted_delta_nll=fit['bias']+result['quality'],predicted_linear_ms=result['latency_ms'],solver={k:v for k,v in result.items() if k!='indices'})
        for fraction in rules['inverse_latency_fractions']:
            cap=float(t[:,0].sum())*fraction;result=solve(q,t,cap,inverse=True)
            add(f'inverse_{scenario}_r{fraction:g}',mapping(result['indices'],phases),scenario,'exact inverse',linear_latency_cap_ms=cap,
                predicted_delta_nll=fit['bias']+result['quality'],predicted_linear_ms=result['latency_ms'],solver={k:v for k,v in result.items() if k!='indices'})
        # Original DP uses fused phase-pair choices and the original quality grid.
        modules=[]
        for i,name in enumerate(names):
            choices=[]
            for pre in range(7):
                for dec in range(7):
                    if len(phases)==1 and pre!=dec:continue
                    quality=q[i,pre]+(q[len(names)+i,dec] if len(phases)==2 else 0.)
                    latency=t[i,pre]+(t[len(names)+i,dec] if len(phases)==2 else 0.)
                    choices.append(Action(methods[pre],methods[dec],float(quality),float(latency)))
            modules.append(ModuleChoices(name,tuple(choices)))
        maximum=sum(max(a.quality_cost for a in m.actions) for m in modules)
        for budget in rules['quality_budgets']:
            start=time.perf_counter();solution=solve_budget(modules,budget-fit['bias'],max_quality=maximum);seconds=time.perf_counter()-start
            maps={n:{'prefill_method':a.prefill_method,'decode_method':a.decode_method} for n,a in zip(names,solution.actions,strict=True)}
            add(f'dp_{scenario}_q{budget:g}',maps,scenario,'original DP',quality_budget=budget,
                predicted_delta_nll=fit['bias']+solution.quality_cost,predicted_linear_ms=solution.latency_ms,solver_seconds=seconds)
        # Search-only scaling uses repeated real 8B action vectors, explicitly not
        # a claim about the 32B model's sensitivities, accuracy, or E2E latency.
        for layers in [36,64,80]:
            count=layers*4*len(phases);indices=np.arange(count)%len(q)
            result=solve(q[indices],t[indices],(.01-fit['bias'])*layers/36)
            scaling.append({'scenario':scenario,'layers':layers,'phase_modules':count,
                'binary_variables':count*7,'seconds':result['seconds'],'optimal':result['optimal'],
                'mip_gap':result['mip_gap'],'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'scope':'search-only repeated real 8B action table, no larger-model quality or E2E claim'})
    # Independent small knapsacks: both objective orientations checked by exhaustive enumeration.
    checks=[];rng=np.random.default_rng(144779)
    for trial in range(8):
        q=rng.random((4,4));t=rng.random((4,4))*10
        allchoices=np.array(list(itertools.product(range(4),repeat=4)))
        qs=q[np.arange(4),allchoices].sum(1);ts=t[np.arange(4),allchoices].sum(1)
        for inverse in [False,True]:
            cap=float(np.median(ts if inverse else qs));result=solve(q,t,cap,inverse)
            objective=qs if inverse else ts;feasible=(ts if inverse else qs)<=cap+1e-10
            assert abs((result['quality'] if inverse else result['latency_ms'])-objective[feasible].min())<1e-7
            checks.append({'trial':trial,'inverse':inverse,'passed':True})
    write_json(RUN/'results/exact_solver_audit.json',{'passed':True,'exhaustive_checks':checks,'scaling':scaling})
    write_json(RUN/'configs/candidate_pool.json',{'points':points,'policies':catalog,'frozen_before_validation':True,
        'design_sha256':hashlib.sha256((RUN/'configs/design.json').read_bytes()).hexdigest()})
    print(json.dumps({'points':len(points),'policies':len(catalog),'small_checks':len(checks),'scaling':scaling}))


if __name__=='__main__':main()
