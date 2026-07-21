#!/usr/bin/env python3
"""Compare Fisher-reweighted local-error features on frozen vLLM-NLL splits."""
from __future__ import annotations

import csv, importlib.util, json, math, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]; DEBUG=ROOT/'artifacts/debug/049_llama2_prefill_fisher_quality_feature'; PREV=ROOT/'artifacts/debug/047_llama2_prefill_mechanism_quality_debug'; SOURCE=ROOT/'artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat'
sys.path.insert(0,str(PREV/'scripts')); from fit_mechanism_proxy import errors, metric, signals, TYPES, PARTS

def read(p):
    with p.open(newline='') as h:return list(csv.DictReader(h))
def fused_fisher():
    rows=read(DEBUG/'module_fisher.csv'); x={(int(r['layer']),r['part']):float(r['fisher_mean_grad_sq']) for r in rows}; out={}
    for l in range(32):
        for typ,parts in PARTS.items(): out[l,typ]=sum(x[l,p] for p in parts)/len(parts)
    geom=math.exp(sum(math.log(max(v,1e-30)) for v in out.values())/len(out)); return {k:v/geom for k,v in out.items()}
def entries():
    old=json.loads((SOURCE/'policies/prefill_only/manifest.json').read_text()); ol={r['policy_id']:r for r in read(SOURCE/'nll/prefill_only.csv')}; new=json.loads((PREV/'manifest.json').read_text()); nl={r['policy_id']:r for r in read(PREV/'nll.csv')}; e=[]
    for z in old:e.append((z['policy_id'],'old_train' if z['split']=='train' else 'old_holdout',json.loads(Path(z['path']).read_text()),float(ol[z['policy_id']]['target_delta_nll'])))
    for z in new:e.append((z['policy_id'],'mechanism_train' if z['split']=='train' else 'mechanism_holdout',json.loads(Path(z['path']).read_text()),float(nl[z['policy_id']]['delta_nll'])))
    return e
def fit(alpha):
    e=entries(); f=fused_fisher(); qe,se=errors('dense_nvfp4'),errors('sparse_bf16'); qe={k:v*f[k]**alpha for k,v in qe.items()}; se={k:v*f[k]**alpha for k,v in se.items()}
    q=torch.stack([signals(x[2],qe,se)[0] for x in e]); s=torch.stack([signals(x[2],qe,se)[1] for x in e]); y=torch.tensor([x[3] for x in e],dtype=torch.float64); tr=torch.tensor([x[1] in {'old_train','mechanism_train'} for x in e]); scale=(q[tr].sum((1,2))+s[tr].sum((1,2))).mean();q/=scale;s/=scale
    p=[torch.full((4,4),.01,dtype=torch.float64,requires_grad=True),torch.full((4,4),.01,dtype=torch.float64,requires_grad=True),torch.full((4,),.01,dtype=torch.float64,requires_grad=True),torch.full((4,),.01,dtype=torch.float64,requires_grad=True)];o=torch.optim.Adam(p,lr=.025)
    for _ in range(5000):
        o.zero_grad();wq,ws,a,c=[torch.relu(z) for z in p];qg,sg=q.sum(2),s.sum(2);pred=(q*wq).sum((1,2))+(s*ws).sum((1,2))+(sg.square()*a).sum(1)+(sg*qg*c).sum(1);(((pred[tr]-y[tr]).square()).mean()+.0001*sum(z.square().mean() for z in p)).backward();o.step()
    result={}
    for g in ('old_holdout','mechanism_holdout'):
        ix=[i for i,x in enumerate(e) if x[1]==g];result[g]=metric([float(y[i]) for i in ix],[float(pred[i].detach()) for i in ix])
    return result
def main():
    output={str(a):fit(a) for a in (0.,.5,1.)};(DEBUG/'fisher_variant_metrics.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
