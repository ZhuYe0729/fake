#!/usr/bin/env python3
"""Measure phase teacher-forced WikiText NLL for one controlled policy."""
from __future__ import annotations
import argparse,csv,gc,json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL=Path('/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf')
PREPARED=Path('/home/agent/wja/project/my/cospaq/fake/artifacts/exports/vllm/baselines/llama2-7b-chat/prepared')
STATE={'dense_nvfp4':'dense_nvfp4','sparse_bf16':'sparse_bf16','sparse_nvfp4':'sparse_nvfp4','w4a16_ours':'marlin_nvfp4'}
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenario',choices=('prefill_only','prefill_decode'),required=True);p.add_argument('--policy',required=True);p.add_argument('--policy-json',type=Path);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output-csv',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--blocks',type=int,default=100);return p.parse_args()
def parent(model,name):
 obj=model
 for part in name.split('.')[:-1]:obj=getattr(obj,part)
 return obj,name.rsplit('.',1)[-1]
def sources(fused):
 base,typ=fused.rsplit('.',1)
 if typ=='qkv_proj':return [base+'.q_proj',base+'.k_proj',base+'.v_proj']
 if typ=='gate_up_proj':return [base+'.gate_proj',base+'.up_proj']
 return [fused]
def install(model,policy,phase):
 saved=[]
 for method,artifact in STATE.items():
  selected=[n for n,e in policy['method_map'].items() if e[f'{phase}_method']==method]
  if not selected:continue
  state=torch.load(PREPARED/artifact/'model.pt',map_location='cpu')['state_dict']
  for fused in selected:
   for name in sources(fused):
    obj,child=parent(model,name);old=getattr(obj,child);new=nn.Linear(old.in_features,old.out_features,bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype);new.weight.data.copy_(state[f'{name}.weight'].to(old.weight));
    if old.bias is not None:new.bias.data.copy_(old.bias.data)
    setattr(obj,child,new);saved.append((obj,child,old))
  del state;gc.collect()
 return saved
def restore(saved):
 for obj,child,old in saved:setattr(obj,child,old)
@torch.inference_mode()
def nll(model,blocks,phase,device,batch):
 loss=tokens=0
 for start in range(0,len(blocks),batch):
  full=blocks[start:start+batch].to(device);ids=full[:,:2048] if phase=='prefill' else full
  logits=model(input_ids=ids,use_cache=False).logits[:,:-1].float()
  labels=ids[:,1:].clone()
  if phase=='decode':labels[:,:2047]=-100
  loss+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum').item());tokens+=int((labels!=-100).sum().item())
 return {'nll':loss/max(tokens,1),'tokens':tokens}
def write(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,row.keys());w.writeheader();w.writerow(row)
def main():
 a=parse();torch.cuda.set_device(a.gpu);device=f'cuda:{a.gpu}';blocks=torch.load(a.output_root/'samples/wikitext_2048_80.pt',map_location='cpu')[:a.blocks];policy=json.loads(a.policy_json.read_text()) if a.policy_json else json.loads((a.output_root/'policies'/a.scenario/f'{a.policy}.json').read_text())
 model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').to(device).eval();dense_pre=nll(model,blocks,'prefill',device,a.batch_size);dense_dec=nll(model,blocks,'decode',device,a.batch_size)
 saved=install(model,policy,'prefill')
 try:pre=nll(model,blocks,'prefill',device,a.batch_size)
 finally:restore(saved);torch.cuda.empty_cache()
 if a.scenario=='prefill_decode':
  saved=install(model,policy,'decode')
  try:dec=nll(model,blocks,'decode',device,a.batch_size)
  finally:restore(saved);torch.cuda.empty_cache()
 else:dec=dense_dec
 dp,dd=pre['nll']-dense_pre['nll'],dec['nll']-dense_dec['nll'];write(a.output_csv,{'policy_id':a.policy,'scenario':a.scenario,'sample_count':len(blocks),'dense_prefill_nll':dense_pre['nll'],'dense_decode_nll':dense_dec['nll'],'prefill_nll':pre['nll'],'decode_nll':dec['nll'],'delta_prefill_nll':dp,'delta_decode_nll':dd,'target_delta_nll':dp+(80*dd if a.scenario=='prefill_decode' else 0.0),'prefill_tokens':pre['tokens'],'decode_tokens':dec['tokens']});print(f'finished {a.scenario}/{a.policy}',flush=True)
if __name__=='__main__':main()
