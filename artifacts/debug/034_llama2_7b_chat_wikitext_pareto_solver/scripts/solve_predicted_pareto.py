#!/usr/bin/env python3
"""Freeze the WikiText proxy and solve predicted fused-module Pareto policies."""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy'
CUTLASS=ROOT/'fake/kernels/cutlass/cutlass_wrapper'
sys.path[:0]=[str(ROOT),str(CUTLASS),str(CUTLASS/'modeling')]
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT,KernelLatencyPredictor # noqa:E402

METHODS=('dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours');PRED={'w4a16_ours':'marlin_nvfp4'};RUNTIME={**{x:x for x in METHODS if x!='w4a16_ours'},'w4a16_ours':'w4a16_ours'}
TYPES=('qkv_proj','o_proj','gate_up_proj','down_proj');SHAPE={'qkv_proj':(12288,4096),'o_proj':(4096,4096),'gate_up_proj':(22016,4096),'down_proj':(4096,11008)};PARTS={'qkv_proj':('q_proj','k_proj','v_proj'),'o_proj':('o_proj',),'gate_up_proj':('gate_proj','up_proj'),'down_proj':('down_proj',)}
SCENARIO={'prefill_only':(8*2048,8,0),'prefill_decode':(16*2048,16,80)}
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenario',choices=tuple(SCENARIO),required=True);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--source-root',type=Path,default=SOURCE);p.add_argument('--budget-bins',type=int,default=1600);p.add_argument('--points',type=int,default=17);return p.parse_args()
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def fused_error():
 rows=read(ROOT/'artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv');out={};alias={'w4a16_ours':'dense_nvfp4'}
 for layer in range(32):
  for typ,parts in PARTS.items():
   for method in METHODS:
    if method=='dense_bf16':out[layer,typ,method]=0.;continue
    values=[float(r['local_rel_mse']) for r in rows if int(r['layer'])==layer and r['module_type'] in parts and r['method']==alias.get(method,method)]
    out[layer,typ,method]=sum(values)/max(len(values),1)
 return out
def feature(policy,phase,err):
 x=torch.zeros((5,4,4),dtype=torch.float64)
 for name,e in policy['method_map'].items():
  layer=int(name.split('.')[2]);typ=name.rsplit('.',1)[-1];m=e[f'{phase}_method'];x[METHODS.index(m),layer//8,TYPES.index(typ)]+=err[layer,typ,m]
 return x
def fit_proxy(source,scenario,err):
 manifest=json.loads((source/'policies'/scenario/'manifest.json').read_text());pol={r['policy_id']:json.loads(Path(r['path']).read_text()) for r in manifest};nll={r['policy_id']:r for r in read(source/'nll'/f'{scenario}.csv')};train=torch.tensor([r['split']=='train' for r in manifest]);xp=torch.stack([feature(pol[r['policy_id']],'prefill',err) for r in manifest]);xd=torch.stack([feature(pol[r['policy_id']],'decode',err) for r in manifest]);y=torch.tensor([float(nll[r['policy_id']]['delta_prefill_nll'])+(80*float(nll[r['policy_id']]['delta_decode_nll']) if scenario=='prefill_decode' else 0.) for r in manifest],dtype=torch.float64);X=xp+(80*xd if scenario=='prefill_decode' else 0.);scale=X[train].sum((1,2,3)).mean().clamp(min=1e-12);X=X/scale
 g=torch.zeros(1,dtype=torch.float64,requires_grad=True);m=torch.zeros(5,dtype=torch.float64,requires_grad=True);b=torch.zeros(4,dtype=torch.float64,requires_grad=True);t=torch.zeros(4,dtype=torch.float64,requires_grad=True);bias=torch.zeros(1,dtype=torch.float64,requires_grad=True);ps=[g,m,b,t,bias];opt=torch.optim.Adam(ps,lr=.03)
 for _ in range(3000):
  opt.zero_grad();coef=torch.nn.functional.softplus(g+m[:,None,None]+b[None,:,None]+t[None,None,:]);pred=bias+(X*coef).sum((1,2,3));loss=((pred[train]-y[train])**2).mean()+.05*sum((v*v).mean() for v in ps[:-1]);loss.backward();opt.step()
 with torch.no_grad():coef=torch.nn.functional.softplus(g+m[:,None,None]+b[None,:,None]+t[None,None,:]);pred=(bias+(X*coef).sum((1,2,3)));hold=~train;rank=lambda z:torch.argsort(torch.argsort(z)).double();a,bv=rank(y[hold]),rank(pred[hold]);spearman=float(((a-a.mean())*(bv-bv.mean())).sum()/torch.sqrt(((a-a.mean())**2).sum()*((bv-bv.mean())**2).sum()))
 return {'coef':coef.detach().tolist(),'scale':float(scale.detach()),'bias':float(bias.detach()),'holdout_spearman':spearman,'training_policies':int(train.sum())}
def modules():
 return [(f'model.layers.{l}.{part}.{typ}',l,typ) for l in range(32) for part,typ in (('self_attn','qkv_proj'),('self_attn','o_proj'),('mlp','gate_up_proj'),('mlp','down_proj'))]
def legal_pairs(scenario):
 if scenario=='prefill_only':return [(m,m) for m in METHODS]
 out=[]
 for p in METHODS:
  for d in METHODS:
   if d=='sparse_nvfp4':continue
   if p==d or {p,d}=={'dense_nvfp4','w4a16_ours'}:out.append((p,d))
 return out
def latency_tables(predictor,scenario):
 mp,md,steps=SCENARIO[scenario];out={}
 for typ,(n,k) in SHAPE.items():
  pre={x.kernel:float(x.latency_ms) for x in predictor.predict(mp,n,k).candidates if x.supported and x.latency_ms is not None};dec={x.kernel:float(x.latency_ms) for x in predictor.predict(md,n,k).candidates if x.supported and x.latency_ms is not None} if steps else {};conv={x.conversion:float(x.latency_ms) for x in predictor.predict_conversion(n,k) if x.supported and x.latency_ms is not None};out[typ]=(pre,dec,conv)
 return out
def candidates(scenario,fit,err):
 predictor=KernelLatencyPredictor(model_root=DEFAULT_MODEL_ROOT,kernels=('dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','marlin_nvfp4'));lat=latency_tables(predictor,scenario);coef=torch.tensor(fit['coef'],dtype=torch.float64);out=[];steps=SCENARIO[scenario][2]
 for index,(name,layer,typ) in enumerate(modules()):
  pre,dec,conv=lat[typ];rows=[]
  for pm,dm in legal_pairs(scenario):
   pk,dk=PRED.get(pm,pm),PRED.get(dm,dm)
   if pk not in pre or (steps and dk not in dec):continue
   needed={('canonical_to_cutlass' if x=='dense_nvfp4' else 'canonical_to_marlin') for x in (pm,dm) if x in {'dense_nvfp4','w4a16_ours'}}
   if not needed.issubset(conv):continue
   q=(err[layer,typ,pm]+(80*err[layer,typ,dm] if steps else 0.))*float(coef[METHODS.index(pm),layer//8,TYPES.index(typ)]) / fit['scale']
   time=pre[pk]+(steps*dec[dk] if steps else 0.)+sum(conv[x] for x in needed)
   rows.append({'module_index':index,'module_name':name,'layer':layer,'module_type':typ,'prefill_method':pm,'decode_method':dm,'quality_cost':q,'latency_ms':time,'prefill_runtime':RUNTIME[pm],'decode_runtime':RUNTIME[dm]})
  if not any(r['prefill_method']=='dense_bf16' and r['decode_method']=='dense_bf16' for r in rows):raise RuntimeError(f'missing dense candidate {name}')
  out.append(rows)
 return out
def solve(groups,budget,bins,maxq):
 scale=bins/max(maxq,1e-12);dp={0:(0.,())}
 for rows in groups:
  nxt={}
  for used,(time,choices) in dp.items():
   for i,r in enumerate(rows):
    q=0 if r['quality_cost']<=0 else max(1,math.ceil(r['quality_cost']*scale));new=used+q
    if new>budget:continue
    value=time+r['latency_ms'];old=nxt.get(new)
    if old is None or value<old[0]:nxt[new]=(value,choices+(i,))
  best=math.inf;dp={q:v for q,v in sorted(nxt.items()) if not (v[0]>=best or (best:=v[0]) is None)}
 if not dp:raise RuntimeError('no feasible state')
 q,(time,choices)=min(dp.items(),key=lambda kv:(kv[1][0],kv[0]));return q,time,choices,scale
def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
def main():
 a=parse();outroot=a.output_root/a.scenario;err=fused_error();fit=fit_proxy(a.source_root,a.scenario,err);groups=candidates(a.scenario,fit,err);maxq=sum(max(r['quality_cost'] for r in g) for g in groups);ratios=[0.]+[10**(-3+i*(3/max(a.points-2,1))) for i in range(a.points-1)];points=[];seen=set()
 for idx,ratio in enumerate(ratios):
  q,time,choices,scale=solve(groups,round(ratio*a.budget_bins),a.budget_bins,maxq);selected=[dict(g[i]) for g,i in zip(groups,choices)];actualq=sum(x['quality_cost'] for x in selected);key=tuple((x['prefill_method'],x['decode_method']) for x in selected)
  if key in seen:continue
  seen.add(key);counts={m:sum(x['prefill_method']==m for x in selected) for m in METHODS};point={'point_index':len(points),'quality_ratio':ratio,'predicted_quality_cost':actualq,'raw_predicted_linear_ms':time,'raw_linear_speedup_vs_dense':0.,**{f'prefill_count_{m}':counts[m] for m in METHODS}};points.append(point)
  method_map={x['module_name']:{'prefill_method':x['prefill_runtime'],'decode_method':x['decode_runtime']} for x in selected};policy={'default_prefill_method':'dense_bf16','default_decode_method':'dense_bf16','modules_to_not_convert':['lm_head'],'method_map':method_map};name=f"point_{point['point_index']:03d}";path=outroot/'pareto/policies'/f'{name}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(policy,indent=2,sort_keys=True)+'\n');write_csv(outroot/'pareto/policies'/f'{name}.csv',selected);point['policy_json']=str(path)
 dense=next(p['raw_predicted_linear_ms'] for p in points if p['quality_ratio']==0.0);[p.update({'raw_linear_speedup_vs_dense':dense/p['raw_predicted_linear_ms']}) for p in points];write_csv(outroot/'pareto/pareto_points.csv',points);(outroot/'frozen_proxy.json').write_text(json.dumps({'scenario':a.scenario,**fit,'quality_intercept_excluded_from_optimizer':True},indent=2)+'\n');(outroot/'pareto/metadata.json').write_text(json.dumps({'scenario':a.scenario,'modules':len(groups),'candidate_actions':sum(map(len,groups)),'budget_bins':a.budget_bins,'points':len(points),'predicted_only':True,'speed_metric':'raw KernelLatencyPredictor linear latency'},indent=2)+'\n');print(f'wrote {len(points)} predicted points')
if __name__=='__main__':main()
