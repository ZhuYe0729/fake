#!/usr/bin/env python3
"""Schedule Llama3.1 policies through the shared real-vLLM evaluator."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; DEBUG=ROOT/"artifacts/debug/043_llama31_prefill_only_vllm_runtime_quality"; EVAL=ROOT/"artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality/scripts/evaluate_policy.py"; TASKS=("wikitext","winogrande","arc_easy","arc_challenge","mmlu")
def main():
 p=argparse.ArgumentParser();p.add_argument("--gpus",default="1,2,3,4,5,6");p.add_argument("--selection");p.add_argument("--limit",type=int);p.add_argument("--audit",action="store_true");p.add_argument("--force",action="store_true");a=p.parse_args()
 m=json.loads((DEBUG/"manifest/policies.json").read_text()); wanted=set(a.selection.split(",")) if a.selection else None; policies=[x for x in m["policies"] if wanted is None or x["label"] in wanted]; profile="full" if a.limit is None else f"limit_{a.limit}"
 def done(x):
  try:return all(json.loads((DEBUG/"results"/x["label"]/t/profile/"result.json").read_text()).get("metrics") for t in TASKS)
  except (OSError,ValueError):return False
 jobs=[x for x in policies if a.force or not done(x)];gpus=[x for x in a.gpus.split(",") if x];state={"queued":len(jobs),"completed":0,"failed":[],"gpus":gpus,"profile":profile};(DEBUG/"run_state").mkdir(parents=True,exist_ok=True);state_path=DEBUG/"run_state"/f"{profile}.json";state_path.write_text(json.dumps(state,indent=2)+"\n");(DEBUG/"logs"/profile).mkdir(parents=True,exist_ok=True);running={}
 while jobs or running:
  for gpu in gpus:
   if gpu in running or not jobs:continue
   x=jobs.pop(0);cmd=[sys.executable,str(EVAL),"--manifest",str(DEBUG/"manifest/policies.json"),"--output-root",str(DEBUG/"results"),"--policy",x["label"]];
   if a.limit is not None:cmd += ["--limit",str(a.limit)]
   if a.audit:cmd += ["--audit"]
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,TOKENIZERS_PARALLELISM="false");h=(DEBUG/"logs"/profile/f"{x['label']}.log").open("w");running[gpu]=(subprocess.Popen(cmd,cwd=DEBUG,env=env,stdout=h,stderr=subprocess.STDOUT),x,h)
  time.sleep(5)
  for gpu,(proc,x,h) in list(running.items()):
   if proc.poll() is None:continue
   h.close();del running[gpu];state["completed"]+=1
   if proc.returncode!=0 or not done(x):state["failed"].append({"gpu":gpu,"policy":x["label"],"exit_code":proc.returncode})
   state_path.write_text(json.dumps(state,indent=2)+"\n")
 if state["failed"]:raise SystemExit(state["failed"])
if __name__=="__main__":main()
