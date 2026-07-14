#!/usr/bin/env python3
"""Fit a positive phase-local/global quality proxy and frozen-holdout metrics."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
METHODS=('dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours')
TYPES=('qkv_proj','o_proj','gate_up_proj','down_proj')
def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def rank(v):return [1+sum(x<y for x in v)+.5*sum(x==y for x in v) for y in v]
def metrics(a,p):
    e=[x-y for x,y in zip(a,p)];ra,rp=rank(a),rank(p);ma=sum(ra)/len(ra);mp=sum(rp)/len(rp);den=math.sqrt(sum((x-ma)**2 for x in ra)*sum((x-mp)**2 for x in rp))
    return {'mae':sum(abs(x) for x in e)/len(e),'rmse':math.sqrt(sum(x*x for x in e)/len(e)),'spearman':sum((x-ma)*(y-mp) for x,y in zip(ra,rp))/den if den else 0.}
def error_table(phase):
    out={}
    for method in METHODS:
        if method=='dense_bf16':
            for b in range(4):
                for typ in TYPES:out[b,typ,method]=0.
        else:
            for row in read(ROOT/'local_errors'/f'{phase}_{method}.csv'):out[int(row['layer_bucket']),row['fused_type'],method]=float(row['output_rel_mse'])
    return out
def feature(policy,errors,phase):
    x=torch.zeros((5,4,4),dtype=torch.float64)
    for name,item in policy['method_map'].items():
        layer=int(name.split('.')[2]);typ=name.rsplit('.',1)[-1];method=item[f'{phase}_method'];x[METHODS.index(method),layer//8,TYPES.index(typ)]+=errors[layer//8,typ,method]
    return x
def main():
    manifest=json.loads((ROOT/'policies/prefill_decode/manifest.json').read_text());labels={x['policy_id']:x for x in read(ROOT/'nll/prefill_decode.csv')};policies=[json.loads(Path(x['path']).read_text()) for x in manifest]
    pre=torch.stack([feature(x,error_table('prefill'),'prefill') for x in policies]);dec=torch.stack([feature(x,error_table('decode'),'decode') for x in policies]);y=torch.tensor([float(labels[x['policy_id']]['target_delta_nll']) for x in policies],dtype=torch.float64);train=torch.tensor([x['split']=='train' for x in manifest]);scale=(pre[train]+80*dec[train]).sum((1,2,3)).mean().clamp(min=1e-12);X=(pre+80*dec)/scale
    g,m,b,t,bias=[torch.zeros(s,dtype=torch.float64,requires_grad=True) for s in (1,5,4,4,1)];params=[g,m,b,t,bias];opt=torch.optim.Adam(params,lr=.03)
    for _ in range(3000):
        opt.zero_grad();coef=torch.nn.functional.softplus(g+m[:,None,None]+b[None,:,None]+t[None,None,:]);pred=bias+(X*coef).sum((1,2,3));loss=((pred[train]-y[train])**2).mean()+.05*sum((q*q).mean() for q in params[:-1]);loss.backward();opt.step()
    with torch.no_grad():
        coef=torch.nn.functional.softplus(g+m[:,None,None]+b[None,:,None]+t[None,None,:]);pred=(bias+(X*coef).sum((1,2,3))).tolist()
    rows=[{'policy_id':v['policy_id'],'split':v['split'],'actual_delta_nll':float(y[i]),'predicted_delta_nll':pred[i]} for i,v in enumerate(manifest)];out=ROOT/'reports/quality';out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'model':'positive normalized phase-local aggregation + method/bucket/type factors; prefill + 80*decode','sample_blocks':int(labels['p00']['sample_count']),'metrics':{s:metrics([x['actual_delta_nll'] for x in rows if x['split']==s],[x['predicted_delta_nll'] for x in rows if x['split']==s]) for s in ('train','holdout')}}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'model.json').write_text(json.dumps({'feature_scale':float(scale),'global':float(g.detach()),'method':m.detach().tolist(),'bucket':b.detach().tolist(),'type':t.detach().tolist(),'bias':float(bias.detach()),'fit_split':'p00-p53 only'},indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
