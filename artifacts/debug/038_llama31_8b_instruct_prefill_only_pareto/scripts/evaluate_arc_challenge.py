#!/usr/bin/env python3
"""Full ARC-Challenge 0-shot answer likelihood for one Llama3 policy."""
from __future__ import annotations
import argparse,csv,gc,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct')
PREPARED=Path('/home/agent/wja/project/my/cospaq/fake/artifacts/exports/vllm/baselines/llama3.1-8b-instruct/prepared')
STATE={'dense_nvfp4':'dense_nvfp4','sparse_bf16':'sparse_bf16','sparse_nvfp4':'sparse_nvfp4','w4a16_ours':'marlin_nvfp4'}
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy-json',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--batch-size',default='4');return p.parse_args()
def parent(model,name):
 obj=model
 for part in name.split('.')[:-1]:obj=getattr(obj,part)
 return obj,name.rsplit('.',1)[-1]
def sources(fused):
 base,kind=fused.rsplit('.',1)
 if kind=='qkv_proj':return [base+'.q_proj',base+'.k_proj',base+'.v_proj']
 if kind=='gate_up_proj':return [base+'.gate_proj',base+'.up_proj']
 return [fused]
def install(model,policy):
 for method,artifact in STATE.items():
  selected=[name for name,item in policy['method_map'].items() if item['prefill_method']==method]
  if not selected:continue
  state=torch.load(PREPARED/artifact/'model.pt',map_location='cpu')['state_dict']
  for fused in selected:
   for name in sources(fused):
    obj,child=parent(model,name);getattr(obj,child).weight.data.copy_(state[f'{name}.weight'].to(getattr(obj,child).weight))
  del state;gc.collect()
def main():
 a=parse();torch.cuda.set_device(a.gpu);device=f'cuda:{a.gpu}';policy=json.loads(a.policy_json.read_text())
 model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').to(device).eval();install(model,policy)
 try:
  import lm_eval
  from lm_eval.models.huggingface import HFLM
  lm=HFLM(pretrained=model,tokenizer=str(MODEL),backend='causal',dtype=torch.bfloat16,device=device,batch_size=a.batch_size,trust_remote_code=False)
  result=lm_eval.simple_evaluate(model=lm,tasks=['arc_challenge'],num_fewshot=0,batch_size=a.batch_size,log_samples=False)
  metrics=result['results']['arc_challenge'];row={'label':a.label,'policy_json':str(a.policy_json),'batch_size':a.batch_size,'acc':metrics.get('acc,none'),'acc_norm':metrics.get('acc_norm,none'),'raw_metrics':metrics}
  a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');print(json.dumps(row,sort_keys=True),flush=True)
 finally:
  del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
