"""Original downstream protocols at fixed batches; no speed measurements."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from common import MODEL, RUN, WRAPPER, write_json


def frozen_datasets():
 import datasets
 datasets.disable_caching()
 def load(path,name=None,*args,split=None,**kwargs):
  root=RUN/'data/likelihood_datasets';namespace=path.replace('/','___').lower()
  roots=[p for p in root.iterdir() if p.is_dir() and p.name.lower()==namespace]
  assert len(roots)==1,(path,name)
  files=list((roots[0]/(name or 'default')).rglob('*.arrow'))
  files.extend((RUN/'data/likelihood_auxiliary_datasets'/roots[0].name/(name or 'default')).rglob('*.arrow'))
  tables={}
  for part in ['train','dev','test','validation']:
   matches=[p for p in files if p.stem.endswith('-'+part)];assert len(matches)<=1,(path,name,part)
   if matches:tables[part]=datasets.Dataset.from_file(str(matches[0]))
  assert tables,(path,name)
  result=datasets.DatasetDict(tables)
  return result[split] if split is not None else result
 datasets.load_dataset=load


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--attempt',required=True);args=parser.parse_args()
 design=json.loads((RUN/'configs/design.json').read_text());job=next(j for j in design['jobs'] if j['id']==args.job)
 out=RUN/'results/attempts'/args.job/args.attempt;out.mkdir(parents=True,exist_ok=False)
 pid=job['policy'];batch_size=job['batch'];checkpoint=RUN/f'data/work/checkpoints/{pid}'
 policy=json.loads((RUN/f'configs/policies/{pid}.json').read_text())
 mapping=json.loads((checkpoint/'config.json').read_text())['quantization_config']['method_map']
 assert mapping==policy['method_map']
 assert hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()==design['policies'][pid]['map_sha256']
 os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_FLASHINFER_SAMPLER='0',PHASE_HETERO_TRACE='0',
  CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES='16')
 for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
  os.environ[key]=str(WRAPPER)
 from telemetry import nvml,HANDLE,PHYSICAL_GPU
 import torch
 uuid=nvml.nvmlDeviceGetUUID(HANDLE)
 assert str(torch.cuda.get_device_properties(0).uuid)==str(uuid).removeprefix('GPU-')
 from runtime_extensions import install
 phase=install()
 from graph_bindings import install as bindings
 bindings()
 from arena_runtime import ArenaRuntime
 runtime=ArenaRuntime(phase);started=time.perf_counter()
 metadata={'job':job,'policy_map_sha256':design['policies'][pid]['map_sha256'],
  'gpu':PHYSICAL_GPU,'gpu_uuid':uuid,'batch':batch_size,'quality_only':True}
 model=None
 try:
  if job['kind']=='generation':
   from vllm import LLM,SamplingParams
   from vllm.inputs import TokensPrompt
   source=RUN/f"data/generation_{job['task']}.jsonl"
   assert hashlib.sha256(source.read_bytes()).hexdigest()==design['input_sha256'][source.name]
   questions=[json.loads(line) for line in source.read_text().splitlines()][job['begin']:job['end']]
   assert len(questions)==job['end']-job['begin']
   model=LLM(model=str(checkpoint),dtype='bfloat16',enforce_eager=False,
    compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[batch_size]},
    enable_prefix_caching=False,enable_chunked_prefill=False,max_model_len=4096,max_num_seqs=batch_size,
    max_num_batched_tokens=32768,gpu_memory_utilization=.90,kv_cache_memory_bytes=6*1024**3)
   runtime.activate();sampling=SamplingParams(temperature=0.,max_tokens=256)
   answers=out/'answers.jsonl'
   with answers.open('w') as handle:
    for start in range(0,len(questions),batch_size):
     rows=questions[start:start+batch_size]
     if phase.current_phase()=='decode':runtime.restore();runtime.wait_prefill()
     outputs=model.generate([TokensPrompt(prompt_token_ids=r['prompt_token_ids']) for r in rows],sampling,use_tqdm=False)
     assert len(outputs)==len(rows)
     for question,response in zip(rows,outputs,strict=True):
      value={'question_id':question['question_id'],'reference':question['reference'],
       'prediction':response.outputs[0].text,'generated_token_ids':list(response.outputs[0].token_ids),
       'prompt_tokens':len(question['prompt_token_ids'])}
      assert len(value['generated_token_ids'])<=256
      handle.write(json.dumps(value,ensure_ascii=False)+'\n')
     handle.flush()
     print(args.job,start+len(rows),'/',len(questions),flush=True)
   metadata.update(complete=True,samples=len(questions),answers=str(answers.relative_to(RUN)),
    input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),max_new_tokens=256,
    chat_template='native Qwen3, enable_thinking=False',elapsed_seconds=time.perf_counter()-started)
   write_json(out/'result.json',metadata)
  else:
   frozen_datasets()
   from lm_eval import simple_evaluate
   from lm_eval.models.vllm_causallms import VLLM
   from lm_eval.tasks import TaskManager
   os.environ['CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES']='4'
   lm=VLLM(pretrained=str(checkpoint),tokenizer=str(MODEL),dtype='bfloat16',batch_size=batch_size,
    max_model_len=2048,tensor_parallel_size=1,enforce_eager=True,enable_prefix_caching=False,
    enable_chunked_prefill=False,gpu_memory_utilization=.9,max_num_seqs=batch_size,
    skip_mm_profiling=True,kv_cache_memory_bytes=3*1024**3)
   model=lm.model;runtime.activate();runtime.active=False
   result=simple_evaluate(model=lm,tasks=[job['task']],num_fewshot=0,batch_size=batch_size,
    limit=None,log_samples=True,random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0,task_manager=TaskManager())
   assert result is not None
   selected=result.get('results',{}).get(job['task']) or result.get('groups',{}).get(job['task']) or {}
   metric=selected[job['metric']];assert math.isfinite(float(metric))
   assert sum(r['effective'] for r in result['n-samples'].values())==job['samples']
   assert phase.current_phase()=='prefill' and not runtime.scheduled and runtime.generations==0
   metadata.update(complete=True,samples=job['samples'],metric=job['metric'],metric_value=metric,
    num_fewshot=0,limit=None,max_model_len=2048,raw_lm_eval=result,elapsed_seconds=time.perf_counter()-started)
   write_json(out/'result.json',metadata)
   print(args.job,'metric',metric,flush=True)
 finally:
  if model is not None:model.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
