"""Freeze fixed controls and shape-conditioned policies without test feedback."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from common import RUN, write_json
from daso.optimization import Action, ModuleChoices, solve_budget

KINDS=['qkv_proj','o_proj','gate_up_proj','down_proj']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixed-only',action='store_true');args=parser.parse_args()
    design=json.loads((RUN/'configs/design.json').read_text())
    source=RUN.parent/'002'
    names=list(json.loads((source/'configs/policies/uniform_dense_bf16.json').read_text())['method_map'])
    signatures={};policies={};points=[]

    def add(label,mapping,workload,family,**metadata):
        signature=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
        policy_id=signatures.setdefault(signature,label)
        if policy_id not in policies:
            policy={'policy_id':policy_id,'scenario':workload['scenario'],
                'default_prefill_method':'dense_bf16','default_decode_method':'dense_bf16',
                'modules_to_not_convert':['lm_head'],'method_map':mapping}
            write_json(RUN/f'configs/policies/{policy_id}.json',policy)
            policies[policy_id]={'signature':signature,'workloads':[]}
        if workload['id'] not in policies[policy_id]['workloads']:policies[policy_id]['workloads'].append(workload['id'])
        points.append({'label':label,'policy_id':policy_id,'family':family,'workload':workload['id'],**metadata})

    def uniform(pre,dec):return {n:{'prefill_method':pre,'decode_method':dec} for n in names}
    for method in design['uniform_methods']:
        for w in design['workloads']:add('uniform_'+method,uniform(method,method),w,'uniform')
    for old in ['daso8_prefill_only_02','daso8_prefill_decode_01','daso5_measured_prefill_decode_04']:
        p=json.loads((source/f'configs/policies/{old}.json').read_text())
        for w in design['workloads']:
            if w['scenario']==p['scenario']:
                add('transfer_'+old,p['method_map'],w,'DASO fixed transfer',source_policy=old)
    for pre in ['dense_nvfp4','sq_int8']:
        for w in design['workloads']:
            if w['scenario']=='prefill_decode':add('phase_'+pre+'_w4a16_ours',uniform(pre,'w4a16_ours'),w,'phase-uniform')
    for scenario,index in [('prefill_only','02'),('prefill_decode','01')]:
        old=f'hawq_fp4_{scenario}_{index}'
        p=json.loads((RUN.parent/'001'/f'configs/policies/{old}.json').read_text())
        for w in design['workloads']:
            if w['scenario']==scenario:add('transfer_'+old,p['method_map'],w,'HAWQ FP4 fixed transfer',source_policy=old)
    if not args.fixed_only:
        profile={(r['module_type'],r['method'],int(r['m'])):float(r['latency_ms'])
                 for r in csv.DictReader((RUN/'results/action_profile.csv').open())}
        errors={(r['module'],r['phase'],r['method']):float(r['output_rel_mse'])
                for r in csv.DictReader((RUN/'data/local_errors.csv').open())}
        traces=json.loads((RUN/'reference/hessian_trace_blockwise_p16_b4_s256.json').read_text())
        assert traces['complete'] and len(traces['records'])==252
        hessian={r['module']:r['mean_eigenvalue'] for r in traces['records']}
        sensitivity=defaultdict(float)
        for file in ['weight_quantization_errors.csv','marlin_int4_weight_errors.csv']:
            for r in csv.DictReader((RUN/'reference'/file).open()):
                name=r['module'];part=name.rsplit('.',1)[-1]
                kind='qkv_proj' if part in ['q_proj','k_proj','v_proj'] else 'gate_up_proj' if part in ['gate_proj','up_proj'] else part
                fused=name.rsplit('.',1)[0]+'.'+kind
                sensitivity[fused,r['method']]+=hessian[name]*float(r['weight_error_squared'])
        for w in design['workloads']:
            fit=json.loads((RUN/f'reference/quality_model_{w["scenario"]}.json').read_text())
            phases=['prefill'] if w['scenario']=='prefill_only' else ['prefill','decode']
            def cost(name,method,phase):
                m=w['batch']*w['input'] if phase=='prefill' else w['batch']
                return profile[name.rsplit('.',1)[-1],method,m]*(1 if phase=='prefill' else w['output']-1)
            modules=[]
            for name in names:
                bucket=int(name.split('.')[2])*4//36;kind=KINDS.index(name.rsplit('.',1)[-1]);actions=[]
                for pre in design['methods']:
                    for dec in design['methods']:
                        if len(phases)==1 and pre!=dec:continue
                        quality=0.;latency=0.
                        for pi,phase in enumerate(phases):
                            method=pre if pi==0 else dec
                            if method!='dense_bf16':
                                quality+=errors[name,phase,method]*fit['coefficients'][pi][design['methods'].index(method)][bucket][kind]/fit['feature_scale']
                            latency+=cost(name,method,phase)
                        actions.append(Action(pre,dec,quality,latency))
                modules.append(ModuleChoices(name,tuple(actions)))
            maximum=sum(max(a.quality_cost for a in m.actions) for m in modules)
            for budget in design['quality_budgets']:
                solution=solve_budget(modules,budget-fit['bias'],max_quality=maximum)
                mapping={n:{'prefill_method':a.prefill_method,'decode_method':a.decode_method}
                         for n,a in zip(names,solution.actions,strict=True)}
                add(f'daso_{w["id"]}_q{budget:g}',mapping,w,'DASO workload adapted',quality_budget=budget,
                    predicted_delta_nll=fit['bias']+solution.quality_cost,predicted_linear_ms=solution.latency_ms)
                methods=['marlin_int4','int8','dense_bf16']
                times=np.array([[sum(cost(n,m,p) for p in phases) for m in methods] for n in names])
                scores=np.array([[sensitivity[n,m] for m in methods] for n in names])
                minimum=float(times.min(1).sum());limit=max(solution.latency_ms,minimum)+1e-7
                matrix=lil_matrix((len(names)+1,len(names)*3))
                for i in range(len(names)):matrix[i,i*3:(i+1)*3]=1
                matrix[-1]=times.reshape(-1)
                result=milp(scores.reshape(-1)/max(float(abs(scores).max()),1e-12),
                    integrality=np.ones(scores.size),bounds=Bounds(0,1),
                    constraints=LinearConstraint(matrix.tocsr(),np.r_[np.ones(len(names)),-np.inf],np.r_[np.ones(len(names)),limit]),
                    options={'mip_rel_gap':0.0})
                assert result.success,result.message
                choice=np.rint(result.x).astype(int).reshape(-1,3)
                assert np.all(choice.sum(1)==1) and float((choice*times).sum())<=limit+1e-6
                mapping={n:{'prefill_method':methods[j],'decode_method':methods[j]} for n,j in zip(names,choice.argmax(1),strict=True)}
                add(f'hawq_{w["id"]}_q{budget:g}',mapping,w,'HAWQ workload adapted',paired_daso_budget=budget,
                    requested_latency_limit_ms=solution.latency_ms,latency_limit_ms=limit,
                    infeasible_requested_limit=solution.latency_ms<minimum-1e-7,
                    predicted_linear_ms=float((choice*times).sum()),hessian_sensitivity=float((choice*scores).sum()))
    frozen={'points':points,'policies':policies,'complete_design':not args.fixed_only,
        'rules':'fixed workload/budget sweep; no test feedback; global method-map deduplication',
        'design_sha256':hashlib.sha256((RUN/'configs/design.json').read_bytes()).hexdigest()}
    target=RUN/('configs/fixed_points.json' if args.fixed_only else 'configs/evaluation_points.json')
    write_json(target,frozen)
    print(json.dumps({'points':len(points),'unique_policies':len(policies),'complete_design':not args.fixed_only}))


if __name__=='__main__':main()
