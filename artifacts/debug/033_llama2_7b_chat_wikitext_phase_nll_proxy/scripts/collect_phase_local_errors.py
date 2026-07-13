#!/usr/bin/env python3
"""Collect WikiText phase-specific local output errors for one compression method."""
from __future__ import annotations
import argparse,csv,gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL=Path('/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf')
PREPARED=Path('/home/agent/wja/project/my/cospaq/fake/artifacts/exports/vllm/baselines/llama2-7b-chat/prepared')
ARTIFACT={'dense_nvfp4':'dense_nvfp4','sparse_bf16':'sparse_bf16','sparse_nvfp4':'sparse_nvfp4','w4a16_ours':'marlin_nvfp4'}
LINEARS={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('prefill','decode'),required=True);p.add_argument('--method',choices=tuple(ARTIFACT),required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--blocks',type=int,default=16);p.add_argument('--max-modules',type=int,default=0);p.add_argument('--module-chunk-size',type=int,default=16);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]);return p.parse_args()
def bucket(name):return int(name.split('.')[2])//8
def fused_type(name):
 t=name.rsplit('.',1)[-1]
 return 'qkv_proj' if t in {'q_proj','k_proj','v_proj'} else 'gate_up_proj' if t in {'gate_proj','up_proj'} else t
def main():
 a=parse();torch.cuda.set_device(a.gpu);device=f'cuda:{a.gpu}';blocks=torch.load(a.output_root/'samples/wikitext_2048_80.pt',map_location='cpu')[:a.blocks];state=torch.load(PREPARED/ARTIFACT[a.method]/'model.pt',map_location='cpu')['state_dict'];model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').to(device).eval();acc={}
 def hook(name):
  def fn(module,inputs,output):
   if not inputs or not isinstance(output,torch.Tensor):return
   x=inputs[0];y=output
   if a.phase=='decode':x,y=x[:,-80:],y[:,-80:]
   key=(bucket(name),fused_type(name));row=acc.setdefault(key,{'sse':0.,'ref':0.,'count':0})
   weight=state[f'{name}.weight'].to(device=x.device,dtype=x.dtype,non_blocking=True);hat=F.linear(x,weight,module.bias);err=(hat.float()-y.float());row['sse']+=float(err.square().sum().item());row['ref']+=float(y.float().square().sum().item());row['count']+=int(y.numel());del weight,hat,err
  return fn
 modules=[(n,m) for n,m in model.named_modules() if n.rsplit('.',1)[-1] in LINEARS and f'{n}.weight' in state]
 if a.max_modules:modules=modules[:a.max_modules]
 print(f'collecting {len(modules)} modules on {a.blocks} blocks',flush=True)
 for start in range(0,len(modules),a.module_chunk_size):
  chunk=modules[start:start+a.module_chunk_size];handles=[m.register_forward_hook(hook(n)) for n,m in chunk]
  try:
   with torch.inference_mode():
    for block in blocks:
     ids=(block[:2048] if a.phase=='prefill' else block).unsqueeze(0).to(device);model(input_ids=ids,use_cache=False)
  finally:
   for h in handles:h.remove()
   torch.cuda.empty_cache();gc.collect()
 rows=[{'phase':a.phase,'method':a.method,'layer_bucket':b,'fused_type':t,'blocks':a.blocks,'output_rel_mse':v['sse']/max(v['ref'],1e-12),'output_mse':v['sse']/max(v['count'],1),'output_count':v['count']} for (b,t),v in sorted(acc.items())]
 out=a.output_root/'local_errors'/f'{a.phase}_{a.method}.csv';out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
 print(out)
 del model,state;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
