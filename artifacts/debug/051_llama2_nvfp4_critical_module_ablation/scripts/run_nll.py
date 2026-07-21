#!/usr/bin/env python3
"""Run leave-one-module real-vLLM NLL ablations, one process per GPU."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];DEBUG=ROOT/'artifacts/debug/051_llama2_nvfp4_critical_module_ablation';SRC=ROOT/'artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat';MODEL=Path('/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf');VLLM=Path('/home/agent/wja/project/my/cospaq/test/vllm');CUTLASS=ROOT/'fake/kernels/cutlass/cutlass_wrapper'
def done(item):
 try:
  p=json.loads((DEBUG/'results'/f"{item['policy_id']}.json").read_text());return len(p['blocks'])==100 and p['runtime']['policy_sha256']==item['sha256']
 except: return False
def one(item):
 policy=Path(item['path']);ckpt=Path('/tmp/cospaq_critical_051')/item['policy_id'];output=DEBUG/'results'/f"{item['policy_id']}.json";log=DEBUG/'logs'/f"{item['policy_id']}.log";log.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError(output)
 cmd=[sys.executable,str(VLLM/'artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py'),'--model-path',str(MODEL),'--policy-json',str(policy),'--output-dir',str(ckpt),'--cutlass-wrapper-path',str(CUTLASS)]
 try:
  with log.open('w') as h:
   subprocess.run(cmd,check=True,stdout=h,stderr=subprocess.STDOUT);subprocess.run([sys.executable,str(SRC.parent/'scripts/evaluate_runtime_prefill_nll.py'),'--checkpoint',str(ckpt),'--tokenizer',str(MODEL),'--samples',str(SRC/'samples/wikitext_2048_targets.pt'),'--output',str(output),'--label',item['policy_id'],'--policy-json',str(policy),'--phase-hetero','--blocks','100'],check=True,stdout=h,stderr=subprocess.STDOUT)
 finally:shutil.rmtree(ckpt,ignore_errors=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--gpus',default='1');a.add_argument('--one');x=a.parse_args();m=json.loads((DEBUG/'manifest.json').read_text());d={z['policy_id']:z for z in m}
 if x.one:one(d[x.one]);return
 jobs=[z for z in m if not done(z)];workers={}
 while jobs or workers:
  for g in x.gpus.split(','):
   if g not in workers and jobs:
    z=jobs.pop(0);workers[g]=(z,subprocess.Popen([sys.executable,__file__,'--one',z['policy_id']],env=dict(os.environ,CUDA_VISIBLE_DEVICES=g)))
  time.sleep(2)
  for g,(z,p) in list(workers.items()):
   if p.poll() is not None:
    del workers[g]
    if not done(z):raise RuntimeError(f'failed {z["policy_id"]}')
if __name__=='__main__':main()
