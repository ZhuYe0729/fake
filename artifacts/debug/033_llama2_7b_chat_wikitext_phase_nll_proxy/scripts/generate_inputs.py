#!/usr/bin/env python3
"""Generate fixed WikiText blocks and controlled phase-policy calibration design."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
QUALITY=ROOT/'artifacts/debug/007_llama2_quality_modeling/scripts'
sys.path.insert(0,str(QUALITY))
from common_quality import QualityConfig, load_calibration_blocks  # type: ignore

METHODS=("dense_bf16","dense_nvfp4","sparse_bf16","sparse_nvfp4","w4a16_ours")
TYPES=("qkv_proj","o_proj","gate_up_proj","down_proj")

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--policies',type=int,default=72);p.add_argument('--blocks',type=int,default=300);p.add_argument('--seed',type=int,default=86);return p.parse_args()
def names():
 out=[]
 for layer in range(32):
  out += [(f'model.layers.{layer}.self_attn.qkv_proj',layer,'qkv_proj'),(f'model.layers.{layer}.self_attn.o_proj',layer,'o_proj'),(f'model.layers.{layer}.mlp.gate_up_proj',layer,'gate_up_proj'),(f'model.layers.{layer}.mlp.down_proj',layer,'down_proj')]
 return out
def methods_for(index, scenario):
 result={}
 for name,layer,typ in names():
  bucket=layer//8
  if index<5: pre=dec=METHODS[index]
  elif index<21: # controlled bucket/type cells
   cell=index-5; selected_bucket=cell//4; selected_type=TYPES[cell%4]; chosen=METHODS[1+(cell%4)]
   pre=chosen if bucket==selected_bucket and typ==selected_type else 'dense_bf16'; dec=pre
  elif index<37: # decode-only controlled cells
   cell=index-21; selected_bucket=cell//4; selected_type=TYPES[cell%4]; chosen=METHODS[1+(cell%4)]
   pre='dense_bf16'; dec=chosen if bucket==selected_bucket and typ==selected_type else 'dense_bf16'
  else: # balanced deterministic mixed policy
   pre=METHODS[(index+3*bucket+2*TYPES.index(typ)+layer)%len(METHODS)]
   dec=METHODS[(2*index+bucket+3*TYPES.index(typ)+layer)%len(METHODS)] if scenario=='prefill_decode' else pre
  # The M=16 decode route has no sparse-NVFP4 kernel.
  if scenario=='prefill_decode' and dec=='sparse_nvfp4': dec='dense_nvfp4'
  result[name]={'prefill_method':pre,'decode_method':dec}
 return result
def main():
 a=args();
 if a.policies!=72: raise ValueError('the controlled design is defined for 72 policies')
 cfg=QualityConfig(calib_samples=a.blocks,seq_len=2129,seed=a.seed,output_root=a.output_root)
 blocks,meta=load_calibration_blocks(cfg)
 if tuple(blocks.shape)!=(a.blocks,2129): raise RuntimeError(f'unexpected WikiText blocks {tuple(blocks.shape)}')
 (a.output_root/'samples').mkdir(parents=True,exist_ok=True);torch.save(blocks.cpu(),a.output_root/'samples/wikitext_2048_80.pt');(a.output_root/'samples/metadata.json').write_text(json.dumps({'blocks':a.blocks,'prefill_tokens':2048,'decode_tokens':80,'source':meta},indent=2,default=str)+'\n')
 for scenario in ('prefill_only','prefill_decode'):
  directory=a.output_root/'policies'/scenario;directory.mkdir(parents=True,exist_ok=True);manifest=[]
  for i in range(72):
   pid=f'p{i:02d}';policy={'policy_id':pid,'scenario':scenario,'policy_kind':'controlled' if i<37 else 'balanced_mixed','default_prefill_method':'dense_bf16','default_decode_method':'dense_bf16','modules_to_not_convert':['lm_head'],'method_map':methods_for(i,scenario)}
   path=directory/f'{pid}.json';path.write_text(json.dumps(policy,indent=2,sort_keys=True)+'\n');manifest.append({'policy_id':pid,'split':'train' if i<54 else 'holdout','path':str(path),'policy_kind':policy['policy_kind']})
  (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('wrote WikiText blocks and 72 policies per scenario')
if __name__=='__main__':main()
