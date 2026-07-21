#!/usr/bin/env python3
"""Export one solved canonical policy, then measure real NLL and E2E speed."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
EXP=Path(os.environ.get("COSPAQ_EXPERIMENT_DIR",ROOT/"artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"))
MODEL=Path(os.environ.get("COSPAQ_MODEL_PATH","/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"))
VLLM=Path(os.environ.get("COSPAQ_VLLM_ROOT","/home/agent/wja/project/my/cospaq/test/vllm"))
CUTLASS=ROOT/"fake/kernels/cutlass/cutlass_wrapper"
EXPORTER=VLLM/"artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
EVALUATOR=ROOT/"artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_runtime_prefill_nll.py"
BENCH=ROOT/"artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py"

def main():
 p=argparse.ArgumentParser();p.add_argument('--point',type=int);p.add_argument('--policy-json',type=Path);p.add_argument('--label');p.add_argument('--runs',type=int,default=5);p.add_argument('--blocks',type=int,default=100);p.add_argument('--tmp-root',type=Path,default=Path('/tmp/cospaq_054_pareto'));a=p.parse_args()
 if (a.point is None) == (a.policy_json is None): p.error('provide exactly one of --point or --policy-json')
 label=a.label or f"point_{a.point:03d}"; policy=a.policy_json or EXP/"pareto/policies"/f"{label}.json"; out=EXP/"pareto/validation"; nll=out/"nll"/f"{label}.json"; runs=out/"speed"/label/"runs"
 if nll.exists() and all((runs/f"measured_{i}.json").exists() for i in range(a.runs)): return
 ckpt=a.tmp_root/label
 if ckpt.exists(): raise FileExistsError(ckpt)
 cmd=[sys.executable,str(EXPORTER),'--model-path',str(MODEL),'--policy-json',str(policy),'--output-dir',str(ckpt),'--cutlass-wrapper-path',str(CUTLASS),'--canonical-sparse-bf16-state',str(EXP/'canonical/prepared/sparse_bf16/model.pt'),'--canonical-sparse-nvfp4-state',str(EXP/'canonical/prepared/sparse_nvfp4/model.pt')]
 try:
  subprocess.run(cmd,check=True)
  nll.parent.mkdir(parents=True,exist_ok=True)
  if not nll.exists(): subprocess.run([sys.executable,str(EVALUATOR),'--checkpoint',str(ckpt),'--tokenizer',str(MODEL),'--samples',str(EXP/'samples/wikitext_2048_targets.pt'),'--output',str(nll),'--label',label,'--policy-json',str(policy),'--phase-hetero','--blocks',str(a.blocks)],check=True)
  runs.mkdir(parents=True,exist_ok=True)
  for name in ['warmup',*[f'measured_{i}' for i in range(a.runs)]]:
   target=runs/f'{name}.json'
   if not target.exists(): subprocess.run([sys.executable,str(BENCH),'--checkpoint',str(ckpt),'--output-json',str(target)],check=True)
 finally: shutil.rmtree(ckpt,ignore_errors=True)
if __name__=='__main__': main()
