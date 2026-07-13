#!/usr/bin/env python3
"""Fit normalized pooled and phase-separated low-dimensional proxy ablations."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import torch
METHODS=('dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours');TYPES=('qkv_proj','o_proj','gate_up_proj','down_proj')
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenario',choices=('prefill_only','prefill_decode'),required=True);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--steps',type=int,default=3000);p.add_argument('--l2',type=float,default=.05);p.add_argument('--phase-local-errors',action='store_true');return p.parse_args()
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def bucket(layer):return layer//8
def errors():
 source=Path('/home/agent/wja/project/my/cospaq/fake/artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv');rows=read(source);out={};mapped={'w4a16_ours':'dense_nvfp4'};parts={'qkv_proj':('q_proj','k_proj','v_proj'),'o_proj':('o_proj',),'gate_up_proj':('gate_proj','up_proj'),'down_proj':('down_proj',)}
 for layer in range(32):
  for typ,need in parts.items():
   for method in METHODS:
    if method=='dense_bf16':out[layer,typ,method]=0.;continue
    values=[float(r['local_rel_mse']) for r in rows if int(r['layer'])==layer and r['module_type'] in need and r['method']==mapped.get(method,method)]
    out[layer,typ,method]=sum(values)/max(len(values),1)
 return out
def phase_errors(root):
 out={}; methods=[m for m in METHODS if m!='dense_bf16']
 for phase in ('prefill','decode'):
  for method in methods:
   for row in read(root/'local_errors'/f'{phase}_{method}.csv'):
    out[phase,int(row['layer_bucket']),row['fused_type'],method]=float(row['output_rel_mse'])
  for b in range(4):
   for t in TYPES:out[phase,b,t,'dense_bf16']=0.
 return out
def x(policy,phase,err,phase_local=False):
 result=torch.zeros((5,4,4),dtype=torch.float64)
 for name,e in policy['method_map'].items():
  layer=int(name.split('.')[2]);typ=name.rsplit('.',1)[-1];method=e[f'{phase}_method'];value=err[phase,bucket(layer),typ,method] if phase_local else err[layer,typ,method];result[METHODS.index(method),bucket(layer),TYPES.index(typ)]+=value
 return result
def rank(v):return [1+sum(a<b for a in v)+.5*sum(a==b for a in v) for b in v]
def metrics(y,p):
 err=[a-b for a,b in zip(y,p)];ra,rp=rank(y),rank(p);ma=sum(ra)/len(ra);mp=sum(rp)/len(rp);den=math.sqrt(sum((a-ma)**2 for a in ra)*sum((b-mp)**2 for b in rp));return {'mae':sum(abs(v) for v in err)/len(err),'rmse':math.sqrt(sum(v*v for v in err)/len(err)),'spearman':sum((a-ma)*(b-mp) for a,b in zip(ra,rp))/den if den else 0.}
def fit(X,y,train,steps,l2):
 # 1 global + method + bucket + fused-type factors; no raw numel scaling.
 g=torch.zeros(1,dtype=torch.float64,requires_grad=True);m=torch.zeros(5,dtype=torch.float64,requires_grad=True);b=torch.zeros(4,dtype=torch.float64,requires_grad=True);t=torch.zeros(4,dtype=torch.float64,requires_grad=True);bias=torch.zeros(1,dtype=torch.float64,requires_grad=True);params=[g,m,b,t,bias];opt=torch.optim.Adam(params,lr=.03)
 scale=X[train].sum((1,2,3)).mean().clamp(min=1e-12);X=X/scale
 for _ in range(steps):
  opt.zero_grad();coef=torch.nn.functional.softplus(g+m[:,None,None]+b[None,:,None]+t[None,None,:]);pred=bias+(X*coef).sum((1,2,3));loss=((pred[train]-y[train])**2).mean()+l2*sum((q*q).mean() for q in params[:-1]);loss.backward();opt.step()
 with torch.no_grad():return (bias+(X*torch.nn.functional.softplus(g+m[:,None,None]+b[None,:,None]+t[None,None,:])).sum((1,2,3))).tolist()
def main():
 a=parse();root=a.output_root;manifest=json.loads((root/'policies'/a.scenario/'manifest.json').read_text());pol={r['policy_id']:json.loads(Path(r['path']).read_text()) for r in manifest};nll={r['policy_id']:r for r in read(root/'nll'/f'{a.scenario}.csv')};err=phase_errors(root) if a.phase_local_errors else errors();train=torch.tensor([r['split']=='train' for r in manifest]);Xp=torch.stack([x(pol[r['policy_id']],'prefill',err,a.phase_local_errors) for r in manifest]);yp=torch.tensor([float(nll[r['policy_id']]['delta_prefill_nll']) for r in manifest],dtype=torch.float64);Xd=torch.stack([x(pol[r['policy_id']],'decode',err,a.phase_local_errors) for r in manifest]);yd=torch.tensor([float(nll[r['policy_id']]['delta_decode_nll']) for r in manifest],dtype=torch.float64)
 pooled=fit(Xp+(80*Xd if a.scenario=='prefill_decode' else 0),yp+(80*yd if a.scenario=='prefill_decode' else 0),train,a.steps,a.l2);pre=fit(Xp,yp,train,a.steps,a.l2);dec=fit(Xd,yd,train,a.steps,a.l2) if a.scenario=='prefill_decode' else [0.]*len(manifest);separate=[p+(80*d if a.scenario=='prefill_decode' else 0) for p,d in zip(pre,dec)];actual=(yp+(80*yd if a.scenario=='prefill_decode' else 0)).tolist();rows=[]
 for i,r in enumerate(manifest):rows.append({'policy_id':r['policy_id'],'split':r['split'],'actual_delta_nll':actual[i],'normalized_pooled':pooled[i],'phase_separated':separate[i],'prefill_prediction':pre[i],'decode_prediction':dec[i]})
 out=root/'reports'/a.scenario/('phase_local_errors' if a.phase_local_errors else '');out.mkdir(parents=True,exist_ok=True)
 with (out/'predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
 summary={'scenario':a.scenario,'features':'phase WikiText local output_rel_mse' if a.phase_local_errors else 'mean fused local_rel_mse; 4 layer buckets; no numel','models':{}}
 for model in ('normalized_pooled','phase_separated'):
  summary['models'][model]={s:metrics([r['actual_delta_nll'] for r in rows if r['split']==s],[r[model] for r in rows if r['split']==s]) for s in ('train','holdout')}
 (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
