"""One frozen engine for both scenarios, measured NLL, and repeated timing."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from common import MODEL, REPO, RUN, WRAPPER, write_json


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--checkpoint',required=True)
 parser.add_argument('--policy',required=True)
 parser.add_argument('--mode',choices=['quality','speed'],required=True)
 parser.add_argument('--tier',type=int,choices=[1,2],required=True)
 parser.add_argument('--round',type=int,default=0)
 parser.add_argument('--label',required=True)
 parser.add_argument('--quality-split',choices=['calibration','validation','test'],default='test')
 args=parser.parse_args()
 design=json.loads((RUN/'configs/design.json').read_text());quality=args.mode=='quality'
 spec=design['policies'][args.policy]
 policy_path=RUN/f'configs/policies/{args.policy}.json'
 assert hashlib.sha256(policy_path.read_bytes()).hexdigest()==spec['map_sha256']
 policy=json.loads(policy_path.read_text())
 exported=json.loads((Path(args.checkpoint)/'phase_hetero_policy.json').read_text())
 assert policy['method_map']==exported['method_map']
 cases=[c for c in design['workloads'] if c['tier']==args.tier and (spec['scenario'] is None or c['scenario']==spec['scenario'])]
 if not quality:
  shift=args.round%len(cases);cases=cases[shift:]+cases[:shift]
 target=RUN/f'results/measurements/{args.label}.json'
 assert not target.exists()
 os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_FLASHINFER_SAMPLER='0',
  PHASE_HETERO_TRACE='0',CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES='8')
 for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
  os.environ[key]=str(WRAPPER)
 from graph_bindings import install as bindings
 bindings()
 from runtime_extensions import install
 phase=install()
 from arena_runtime import ArenaRuntime
 runtime=ArenaRuntime(phase)
 import torch
 from vllm import LLM,SamplingParams
 from vllm.inputs import TokensPrompt
 from telemetry import gpu,placement,nvml,HANDLE
 kwargs={}
 if quality:
  sys.path.insert(0,str(REPO/'daso/workflows/paper/runtime'))
  from teacher_force_processor import TeacherForceBatchProcessor
  kwargs['logits_processors']=[TeacherForceBatchProcessor]
 engine=design['engine'];started=time.perf_counter()
 llm=LLM(model=args.checkpoint,tokenizer=str(MODEL),dtype='bfloat16',enforce_eager=False,
  compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':engine['capture_sizes']},
  enable_prefix_caching=False,enable_chunked_prefill=False,max_model_len=engine['max_model_len'],
  max_num_seqs=engine['max_num_seqs'],max_num_batched_tokens=engine['max_num_batched_tokens'],
  kv_cache_memory_bytes=engine['kv_gib']*1024**3,gpu_memory_utilization=.9,skip_mm_profiling=True,**kwargs)
 assert len(phase._LAYERS)==design['module_count']
 for layer in phase._LAYERS:
  expected=policy['method_map'][layer.prefix]
  assert layer.ph_prefill_method==expected['prefill_method'] and layer.ph_decode_method==expected['decode_method']
 runtime.activate();torch.cuda.synchronize();initialization=(time.perf_counter()-started)*1000
 addresses=runtime.addresses();first={'time':None}
 original=llm.llm_engine.output_processor.process_outputs
 def observe(outputs,*a,**kw):
  if first['time'] is None and any(o.new_token_ids for o in outputs):first['time']=time.perf_counter()
  return original(outputs,*a,**kw)
 llm.llm_engine.output_processor.process_outputs=observe
 results=[]
 try:
  for case in cases:
   cache=llm.llm_engine.vllm_config.cache_config
   required=case['batch']*math.ceil((case['prompt']+case['output'])/cache.block_size)*cache.block_size
   assert cache.num_gpu_blocks*cache.block_size>=required
   print(args.label,case['label'],'begin',flush=True)
   if quality:
    destination=RUN/f"results/quality/{args.quality_split}/{args.policy}/{case['label']}.json"
    if destination.exists():
     previous=json.loads(destination.read_text())
     assert previous['complete'] and previous['sample_sha256']==design['quality_sha256'][args.quality_split]
     assert len(previous['blocks'])==design['quality_blocks']
     continue
    path=Path(design['quality_samples'][args.quality_split]);assert hashlib.sha256(path.read_bytes()).hexdigest()==design['quality_sha256'][args.quality_split]
    data=torch.load(path,weights_only=True);rows=[];phase_checks=[]
    for start in range(0,design['quality_blocks'],case['batch']):
     prompts=[];params=[];captures=[];forced=[]
     for offset,block in enumerate(data[start:start+case['batch']]):
      ids=block[:case['prompt']].tolist();targets=block[case['prompt']:case['prompt']+case['output']].tolist()
      settings={'temperature':0.,'max_tokens':case['output'],'min_tokens':case['output'],'ignore_eos':True,'detokenize':False}
      capture=RUN/f"results/captures/{args.quality_split}/{args.policy}/{case['label']}/{args.label}/{start+offset:04d}.jsonl"
      if case['scenario']=='prefill_only':settings['prompt_logprobs']=1
      else:
       assert not capture.exists()
       settings['extra_args']={'teacher_force_target_ids':targets,'teacher_force_capture_path':str(capture)}
      prompts.append(TokensPrompt(prompt_token_ids=ids));params.append(SamplingParams(**settings));captures.append(capture);forced.append(targets)
     if phase.current_phase()=='decode':runtime.restore();runtime.wait_prefill()
     answers=llm.generate(prompts,params,use_tqdm=False)
     assert len(answers)==len(prompts)
     assert phase.current_phase()=='decode' and len(runtime.scheduled)==design['module_count']
     phase_checks.append({'first_block':start,'batch':len(prompts),'modules_scheduled':len(runtime.scheduled),'phase_after':'decode'})
     for offset,answer in enumerate(answers):
      if case['scenario']=='prefill_only':
       ids=prompts[offset]['prompt_token_ids'];values=[float(answer.prompt_logprobs[j][ids[j]].logprob) for j in range(1,len(ids))]
      else:
       assert answer.outputs[0].token_ids==forced[offset]
       values=[json.loads(line)['logprob'] for line in captures[offset].read_text().splitlines()]
      assert len(values)==(case['prompt']-1 if case['scenario']=='prefill_only' else case['output']) and all(math.isfinite(v) for v in values)
      rows.append({'block':start+offset,'avg_nll':-sum(values)/len(values),'token_count':len(values)})
     print(args.label,case['label'],len(rows),'/' + str(design['quality_blocks']),flush=True)
    assert len(rows)==design['quality_blocks']
    record={'complete':True,'policy':args.policy,'workload':case['label'],'case':case,'blocks':rows,'gpu':gpu(),
     'split':args.quality_split,'sample_sha256':design['quality_sha256'][args.quality_split],
     'policy_sha256':spec['map_sha256'],'phase_checks':phase_checks}
    write_json(destination,record);results.append({'workload':case['label'],'quality_complete':True})
   else:
    path=Path(case['input']);assert hashlib.sha256(path.read_bytes()).hexdigest()==case['input_sha256']
    prompts=[TokensPrompt(prompt_token_ids=t) for t in json.loads(path.read_text())]
    params=SamplingParams(temperature=0.,max_tokens=case['output'],min_tokens=case['output'],ignore_eos=True,detokenize=False)
    samples=[]
    for index in range(design['warmups']+design['iterations']):
     torch.cuda.synchronize();before=gpu();begin_restore=time.perf_counter()
     if phase.current_phase()=='decode':runtime.restore();runtime.wait_prefill()
     torch.cuda.synchronize();restore=(time.perf_counter()-begin_restore)*1000
     first['time']=None;begin=time.perf_counter();answers=llm.generate(prompts,params,use_tqdm=False);torch.cuda.synchronize();end=time.perf_counter()
     tokens=[a.outputs[0].token_ids for a in answers]
     assert phase.current_phase()=='decode' and len(runtime.scheduled)==design['module_count']
     assert len(tokens)==case['batch'] and all(len(t)==case['output'] for t in tokens)
     assert first['time'] is not None and runtime.addresses()==addresses
     samples.append({'iteration':index,'warmup':index<design['warmups'],'e2e_ms':(end-begin)*1000,
      'ttft_ms':(first['time']-begin)*1000,'tpot_ms':(end-first['time'])*1000/(case['output']-1) if case['output']>1 else None,
      'restore_ms':restore,'cycle_ms':restore+(end-begin)*1000,'output_token_ids':tokens,
      'output_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),'gpu_before':before,'gpu_after':gpu()})
    keys=['e2e_ms','ttft_ms','restore_ms','cycle_ms']+(['tpot_ms'] if case['output']>1 else [])
    medians={k:statistics.median(s[k] for s in samples if not s['warmup']) for k in keys}
    results.append({'case':case,'samples':samples,'medians':medians})
    print(args.label,case['label'],medians,flush=True)
  weight_audit={}
  for state in ['decode','prefill']:
   if state=='prefill':runtime.restore();runtime.wait_prefill();torch.cuda.synchronize()
   checked=0
   for layer in phase._LAYERS:
    if layer.ph_prefill_method==layer.ph_decode_method:continue
    for name in getattr(layer,'ph_'+state+'_adapter').names:
     value=getattr(layer,'ph_'+state+'_'+name+('_gpu' if state=='decode' else ''))
     reference=getattr(layer,'ph_'+state+'_cpu')[name].reshape(value.shape)
     assert torch.equal(value.cpu(),reference),(state,layer.prefix,name)
     checked+=1
   weight_audit[state]={'exact_equal':True,'tensors':checked}
  import resource
  host_bytes={state:sum(t.numel()*t.element_size() for layer in phase._LAYERS for t in getattr(layer,'ph_'+state+'_cpu',{}).values()) for state in ['prefill','decode']}
  write_json(target,{'complete':True,'policy':args.policy,'mode':args.mode,'tier':args.tier,'round':args.round,
   'cases':results,'initialization_ms':initialization,'engine':engine,'placement':placement(),
   'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'weight_audit':weight_audit,'host_weight_cache_bytes':host_bytes,
   'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
   'peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
   'nvml_used_after_bytes':nvml.nvmlDeviceGetMemoryInfo(HANDLE).used})
 finally:llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
