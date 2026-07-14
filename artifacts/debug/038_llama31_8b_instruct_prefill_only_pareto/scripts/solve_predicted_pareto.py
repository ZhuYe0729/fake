#!/usr/bin/env python3
"""Solve prefill-only mixed policies with frozen quality and speed surrogates.

The output is a screening set only: every displayed point must later receive
fresh vLLM E2E and 100-block NLL measurements.
"""
from __future__ import annotations
import csv,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS=('dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours')
KERNEL={**{m:m for m in METHODS[:-1]},'w4a16_ours':'marlin_nvfp4'}
TYPES=('qkv_proj','o_proj','gate_up_proj','down_proj')
STEP=.002
def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def pava(y):
    blocks=[]
    for value in y:
        blocks.append([value,1.])
        while len(blocks)>1 and blocks[-2][0]/blocks[-2][1]>blocks[-1][0]/blocks[-1][1]:
            right=blocks.pop();blocks[-1][0]+=right[0];blocks[-1][1]+=right[1]
    return [a/b for a,b in blocks for _ in range(int(b))]
def corrected(train,x):
    pairs=sorted(train);xs=[a for a,b in pairs];ys=pava([b for a,b in pairs])
    if x<=xs[0]:return ys[0]
    if x>=xs[-1]:return ys[-1]
    for i in range(1,len(xs)):
        if x<=xs[i]:return ys[i-1]+(x-xs[i-1])/(xs[i]-xs[i-1])*(ys[i]-ys[i-1])
def main():
    model=json.loads((ROOT/'reports/quality/model.json').read_text());scale=model['feature_scale']
    coefficient=lambda mi,b,t:math.log1p(math.exp(model['global']+model['method'][mi]+model['bucket'][b]+model['type'][t]))/scale
    local={}
    for method in METHODS[1:]:
        for row in read(ROOT/'local_errors'/f'prefill_{method}.csv'):local[int(row['layer_bucket']),row['fused_type'],method]=float(row['output_rel_mse'])
    for b in range(4):
        for t in TYPES:local[b,t,'dense_bf16']=0.
    latency={(r['module_name'],r['kernel']):float(r['latency_ms']) for r in read(ROOT/'action_support.csv') if r['supported']=='True'}
    modules=[]
    for layer in range(32):
        for group,typ in (('self_attn','qkv_proj'),('self_attn','o_proj'),('mlp','gate_up_proj'),('mlp','down_proj')):
            name=f'model.layers.{layer}.{group}.{typ}';b=layer//8;t=TYPES.index(typ);opts=[]
            for mi,method in enumerate(METHODS):
                q=local[b,typ,method]*coefficient(mi,b,t)
                # A non-dense action must never become free merely because it
                # is below the DP quantization step: that would corrupt the
                # exact dense-quality endpoint.
                qbin=0 if q == 0. else max(1,int(math.ceil(q/STEP)))
                opts.append((qbin,q,latency[name,KERNEL[method]],method))
            modules.append((name,opts))
    max_bin=850;dp=[math.inf]*(max_bin+1);dp[0]=0.;back=[]
    for _,options in modules:
        nxt=[math.inf]*(max_bin+1);choice=[None]*(max_bin+1)
        for old,value in enumerate(dp):
            if not math.isfinite(value):continue
            for oi,(qb,_,cost,_) in enumerate(options):
                new=min(max_bin,old+qb);candidate=value+cost
                if candidate<nxt[new]:nxt[new]=candidate;choice[new]=(old,oi)
        dp=nxt;back.append(choice)
    # Make each requested budget use the fastest feasible DP state and retain
    # only unique assignments; uniform policies are included as explicit refs.
    requested=[0,.01,.02,.04,.06,.08,.12,.18,.25,.35,.5,.7,1.,1.3,1.7]
    calibration=read(ROOT/'speed_calibration/calibration.csv');train=[(float(r['raw_predicted_linear_ms']),float(r['e2e_median_ms'])) for r in calibration if r['split']=='train']
    rows=[];seen=set();policy_dir=ROOT/'pareto/policies';policy_dir.mkdir(parents=True,exist_ok=True)
    for budget in requested:
        limit=min(max_bin,int(math.floor(budget/STEP)));state=min(range(limit+1),key=lambda x:dp[x])
        picks=[]
        for index in range(len(modules)-1,-1,-1):
            previous,option=back[index][state];picks.append(option);state=previous
        picks=list(reversed(picks));signature=tuple(picks)
        if signature in seen:continue
        seen.add(signature);method_map={name:{'prefill_method':opts[pick][3],'decode_method':opts[pick][3]} for (name,opts),pick in zip(modules,picks)}
        raw=sum(opts[pick][2] for (_,opts),pick in zip(modules,picks));quality=sum(opts[pick][1] for (_,opts),pick in zip(modules,picks))+model['bias'];pid=f'point_{len(rows):03d}'
        policy={'policy_id':pid,'scenario':'prefill_only','policy_kind':'predicted_pareto_screening','default_prefill_method':'dense_bf16','default_decode_method':'dense_bf16','modules_to_not_convert':['lm_head'],'method_map':method_map}
        (policy_dir/f'{pid}.json').write_text(json.dumps(policy,indent=2,sort_keys=True)+'\n')
        rows.append({'point_index':len(rows),'policy_id':pid,'quality_budget':budget,'predicted_delta_nll':max(0.,quality),'raw_predicted_linear_ms':raw,'corrected_e2e_prediction_ms':corrected(train,raw),**{f'count_{m}':sum(x==m for x in (v['prefill_method'] for v in method_map.values())) for m in METHODS}})
    output=ROOT/'pareto';
    with (output/'predicted_points.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'metadata.json').write_text(json.dumps({'quality_model':'54-train fitted positive local+global model; holdout Spearman 0.967','speed_model':'kernel predictor sum + train-only monotone E2E correction; strict-heldout MAE 11.30ms','quality_step':STEP,'status':'screening only; measured closure TODO'},indent=2)+'\n')
    print(output/'predicted_points.csv')
if __name__=='__main__':main()
