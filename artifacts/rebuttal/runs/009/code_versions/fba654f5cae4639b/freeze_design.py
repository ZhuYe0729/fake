"""Sensitivity-sign controls, identical latency caps, no NLL-based selection."""
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix

RUN=Path(__file__).resolve().parents[1]


def main():
    old=RUN.parent/'003';original=RUN.parent/'001'
    for d in ['configs/policies','data','reference','results','logs','code/compat']:(RUN/d).mkdir(parents=True,exist_ok=True)
    for n in ['common.py','eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py']:shutil.copy2(old/'code'/n,RUN/'code'/n)
    shutil.copytree(old/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    for n in ['activation_scales.pt']:
        (RUN/'data'/n).symlink_to(old/'data'/n)
    (RUN/'data/test_64_2112.pt').symlink_to(RUN.parent/'005/data/test_64_2112.pt')
    # Reuse the audited evaluator while retaining this run's local common/RUN.
    text=(RUN.parent/'005/code/evaluate.py').read_text()
    text=text.replace("choices=['calibration_speed','validation','test','test_speed']","choices=['calibration_speed','test']")
    text=text.replace("(RUN/'configs/selected_policies.json').exists()","(RUN.parent/'005/configs/selected_policies.json').exists()")
    (RUN/'code/evaluate.py').write_text(text)
    design={'sensitivity_transforms':['signed','clip_zero','absolute'],'formats':['marlin_int4','fp4_phase_pair'],
        'paired_daso_budgets':[.01,.07],'scenarios':['prefill_only','prefill_decode'],
        'warmups':2,'iterations':5,'quality_score_batch':8,'kv_bytes':6*1024**3,
        'speed_calibration_seed':144947,'speed_test_seed':144949,
        'scope':'Same original HAWQ weight-error objective and frozen Hessian estimates; sign-handling controls. FP4 follows original run001 precision-pair PTQ adaptation. No original CNN QAT/TVM reproduction claim.',
        'quality':'All frozen variants evaluated on same 64 held-out nonoverlapping test windows after run005 selection is fixed; no choosing a transform from test labels.'}
    (RUN/'configs/design.json').write_text(json.dumps(design,indent=2)+'\n')
    traces=json.loads((original/'results/hessian_trace_blockwise_p16_b4_s256.json').read_text())['records']
    hessian={r['module']:r['mean_eigenvalue'] for r in traces}
    errors=[]
    for n in ['weight_quantization_errors.csv','marlin_int4_weight_errors.csv','fp4_weight_quantization_errors.csv']:
        errors.extend(list(csv.DictReader((original/'results'/n).open())))
    profile={(r['module_type'],r['method'],int(r['m'])):float(r['latency_ms']) for r in csv.DictReader((old/'results/action_profile.csv').open())}
    source=json.loads((old/'configs/evaluation_points.json').read_text())['points']
    template=json.loads((old/'configs/policies/uniform_dense_bf16.json').read_text());names=list(template['method_map'])
    catalog={};points=[];signatures={}

    def add(label,mapping,scenario,family,**meta):
        signature=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest();pid=signatures.setdefault(signature,label)
        if pid not in catalog:
            (RUN/f'configs/policies/{pid}.json').write_text(json.dumps({**template,'policy_id':pid,'scenario':scenario,'method_map':mapping},indent=2)+'\n')
            catalog[pid]={'signature':signature,'scenarios':[]}
        if scenario not in catalog[pid]['scenarios']:catalog[pid]['scenarios'].append(scenario)
        points.append({'label':label,'policy_id':pid,'scenario':scenario,'family':family,**meta})

    for method in ['dense_bf16','int8','sq_int8','fp8']:
        mapping={n:{'prefill_method':method,'decode_method':method} for n in names}
        for scenario in design['scenarios']:add('uniform_'+method,mapping,scenario,'uniform')
    for label,scenario in [('transfer_daso5_measured_prefill_decode_04','prefill_decode'),
                           ('transfer_daso8_prefill_decode_01','prefill_decode'),('transfer_daso8_prefill_only_02','prefill_only')]:
        mapping=json.loads((old/f'configs/policies/{label}.json').read_text())['method_map'];add(label,mapping,scenario,'fixed DASO')
    transforms={'signed':lambda h:h,'clip_zero':lambda h:max(h,0.),'absolute':abs}
    for transform,operation in transforms.items():
        sensitivity=defaultdict(float)
        for r in errors:
            name=r['module'];part=name.rsplit('.',1)[-1]
            kind='qkv_proj' if part in ['q_proj','k_proj','v_proj'] else 'gate_up_proj' if part in ['gate_proj','up_proj'] else part
            fused=name.rsplit('.',1)[0]+'.'+kind;sensitivity[fused,r['method']]+=operation(hessian[name])*float(r['weight_error_squared'])
        for fmt in design['formats']:
            methods=['marlin_int4' if fmt=='marlin_int4' else 'dense_nvfp4','int8','dense_bf16']
            for scenario in design['scenarios']:
                workload='p_b8_s2048' if scenario=='prefill_only' else 'pd_b8_s2048_o64'
                def decode(method):return 'w4a16_ours' if fmt=='fp4_phase_pair' and method=='dense_nvfp4' and scenario=='prefill_decode' else method
                times=np.array([[profile[n.rsplit('.',1)[-1],m,16384]+(63*profile[n.rsplit('.',1)[-1],decode(m),8] if scenario=='prefill_decode' else 0.) for m in methods] for n in names])
                scores=np.array([[sensitivity[n,m] for m in methods] for n in names]);minimum=float(times.min(1).sum())
                for budget in design['paired_daso_budgets']:
                    paired=next(p for p in source if p['family']=='DASO workload adapted' and p['workload']==workload and p['quality_budget']==budget)
                    requested=paired['predicted_linear_ms'];limit=max(requested,minimum)+1e-7
                    matrix=lil_matrix((len(names)+1,len(names)*3))
                    for i in range(len(names)):matrix[i,i*3:(i+1)*3]=1
                    matrix[-1]=times.reshape(-1)
                    result=milp(scores.reshape(-1)/max(float(abs(scores).max()),1e-12),integrality=np.ones(scores.size),bounds=Bounds(0,1),
                        constraints=LinearConstraint(matrix.tocsr(),np.r_[np.ones(len(names)),-np.inf],np.r_[np.ones(len(names)),limit]),options={'mip_rel_gap':0.0})
                    assert result.success,result.message
                    choices=np.rint(result.x).astype(int).reshape(-1,3);assert np.all(choices.sum(1)==1)
                    assert float((choices*times).sum())<=limit+1e-6
                    mapping={n:{'prefill_method':methods[j],'decode_method':decode(methods[j])} for n,j in zip(names,choices.argmax(1),strict=True)}
                    add(f'hawq_{fmt}_{transform}_{scenario}_q{budget:g}',mapping,scenario,'HAWQ robustness',
                        sensitivity_transform=transform,format=fmt,paired_daso_budget=budget,
                        requested_linear_limit_ms=requested,linear_limit_ms=limit,minimum_linear_ms=minimum,
                        infeasible_requested_limit=requested<minimum-1e-7,predicted_linear_ms=float((choices*times).sum()),
                        sensitivity_objective=float((choices*scores).sum()),mip_gap=float(result.mip_gap))
    (RUN/'configs/candidate_pool.json').write_text(json.dumps({'points':points,'policies':catalog,'frozen_before_quality':True},indent=2)+'\n')
    (RUN/'results/hessian_sign_audit.json').write_text(json.dumps({'blocks':len(hessian),'negative_blocks':sum(v<0 for v in hessian.values()),
        'minimum':min(hessian.values()),'maximum':max(hessian.values()),'source_sha256':hashlib.sha256((original/'results/hessian_trace_blockwise_p16_b4_s256.json').read_bytes()).hexdigest(),
        'scope':'Transform each original unfused block before summing fused-module sensitivity; all transforms reported, not selected by measured quality.'},indent=2)+'\n')
    print(json.dumps({'points':len(points),'unique_policies':len(catalog),'negative_blocks':sum(v<0 for v in hessian.values())}))


if __name__=='__main__':main()
