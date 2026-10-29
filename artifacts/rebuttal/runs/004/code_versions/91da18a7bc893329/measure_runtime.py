"""Untraced E2E/cycle timings and separate short CPU/CUDA traces."""
import argparse
import gc
import hashlib
import json
import os
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from common import MODEL,REPO,RUN,WRAPPER,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--policy',required=True);p.add_argument('--mode',choices=['async','sync','resident'],required=True)
    p.add_argument('--profile',action='store_true');args=p.parse_args()
    design=json.loads((RUN/'configs/design.json').read_text())
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    os.environ['PHASE_HETERO_TRACE']='0';os.environ['PHASE_HETERO_SYNC_COPY']='1' if args.mode=='sync' else '0'
    os.environ['CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES']='4'
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:os.environ[key]=str(WRAPPER)
    from runtime_extensions import install
    phase=install()
    import torch
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    started=time.perf_counter();options=design['engine']
    llm=LLM(model=str(args.checkpoint),tokenizer=str(MODEL),dtype='bfloat16',enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=False,
        kv_cache_memory_bytes=design['kv_bytes'],gpu_memory_utilization=.90,skip_mm_profiling=True,**options)
    phase.enable_phase_hetero()
    before_resident=torch.cuda.memory_allocated()
    if args.mode=='resident':
        from resident import install as dual
        dual(phase)
    torch.cuda.synchronize();initialization=time.perf_counter()-started

    def storage_bytes(tensors):
        seen={}
        for t in tensors:
            if isinstance(t,torch.Tensor):
                s=t.untyped_storage();seen[(str(t.device),s.data_ptr())]=s.nbytes()
        return sum(seen.values())

    def memory():
        host=[];gpu=[]
        for layer in phase._LAYERS:
            host.extend(layer.ph_prefill_cpu.values());host.extend(layer.ph_decode_cpu.values())
            for name in layer.ph_prefill_adapter.names:gpu.append(getattr(layer,'ph_prefill_'+name,None))
            for name in layer.ph_decode_adapter.names:gpu.append(getattr(layer,'ph_decode_'+name+'_gpu',None))
        rss=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
        return {'host_weights_bytes':storage_bytes(host),'pinned_weights_bytes':storage_bytes([t for t in host if t.is_pinned()]),
            'gpu_linear_weights_bytes':storage_bytes(gpu),'allocated_bytes':torch.cuda.memory_allocated(),
            'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'rss_bytes':rss,
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}

    pref_bytes=storage_bytes([t for l in phase._LAYERS if l.ph_prefill_method!=l.ph_decode_method for t in l.ph_prefill_cpu.values()])
    dec_bytes=storage_bytes([t for l in phase._LAYERS for t in l.ph_decode_cpu.values()])
    initial_memory=memory()

    def restore():
        start=time.perf_counter()
        if phase.current_phase()=='decode':phase.prepare_next_prefill();phase.wait_for_prefill_ready()
        torch.cuda.synchronize()
        return (time.perf_counter()-start)*1000

    def inputs(w,output):
        g=torch.Generator().manual_seed(144+w['batch']*1000003+w['input'])
        ids=torch.randint(100,30000,(w['batch'],w['input']),generator=g)
        prompts=[TokensPrompt(prompt_token_ids=row.tolist()) for row in ids]
        params=SamplingParams(max_tokens=output,min_tokens=output,ignore_eos=True,temperature=0.,detokenize=False)
        return prompts,params

    def timed(w,output):
        prompts,params=inputs(w,output);rows=[];torch.cuda.reset_peak_memory_stats()
        for i in range(design['warmups']+design['iterations']):
            restoration=restore();start=time.perf_counter()
            answers=llm.generate(prompts,params,use_tqdm=False);torch.cuda.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            tokens=[a.outputs[0].token_ids for a in answers]
            assert len(tokens)==w['batch'] and all(len(x)==output for x in tokens)
            digest=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            rows.append({'iteration':i,'warmup':i<design['warmups'],'generation_ms':elapsed,
                'restore_ms':restoration,'cycle_ms':elapsed+restoration,'output_sha256':digest})
        assert len({r['output_sha256'] for r in rows})==1,'non-deterministic output within a mode'
        measured=[r for r in rows if not r['warmup']]
        return {'output':output,'iterations':rows,**{key:statistics.median(r[key] for r in measured)
            for key in ['generation_ms','restore_ms','cycle_ms']},'memory':memory(),'output_sha256':rows[-1]['output_sha256']}

    try:
        if not args.profile:
            for w in design['workloads']:
                target=RUN/f'results/runtime/{args.policy}/{args.mode}/{w["id"]}.json'
                ttft=timed(w,1);end=timed(w,w['output'])
                write_json(target,{'policy':args.policy,'mode':args.mode,'workload':w,'ttft':ttft,'e2e':end,
                    'tpot_ms':(end['generation_ms']-ttft['generation_ms'])/(w['output']-1),
                    'initial_memory':initial_memory,'initialization_seconds':initialization,
                    'resident_gpu_increment_bytes':initial_memory['allocated_bytes']-before_resident if args.mode=='resident' else None,
                    'theoretical_prefill_restore_bytes':0 if args.mode=='resident' else pref_bytes,
                    'theoretical_decode_transfer_bytes':0 if args.mode=='resident' else dec_bytes,
                    'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'kv_bytes':design['kv_bytes'],
                    'scope':'generation excludes restore; cycle includes synchronized restore. Memory is per-process PyTorch allocator and resident host weight accounting.'})
                print(args.policy,args.mode,w['id'],end['generation_ms'],end['cycle_ms'],flush=True)
        else:
            original=phase.PhaseHeteroMyTestLinearMethod.apply
            def marked(self,layer,x,bias=None):
                with torch.profiler.record_function('DASO_Linear/'+phase.current_phase()+'/'+layer.prefix):return original(self,layer,x,bias)
            phase.PhaseHeteroMyTestLinearMethod.apply=marked
            for w in [design['workloads'][0],design['workloads'][2]]:
                prompts,params=inputs(w,4)
                restore();llm.generate(prompts,params,use_tqdm=False);restore()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                    llm.generate(prompts,params,use_tqdm=False);torch.cuda.synchronize()
                directory=RUN/f'results/profiles/{args.policy}/{args.mode}';directory.mkdir(parents=True,exist_ok=True)
                trace=directory/f'{w["id"]}.json';profile.export_chrome_trace(str(trace))
                events=json.loads(trace.read_text())['traceEvents'];kernels=[];copies=[];names=Counter()
                for e in events:
                    if e.get('ph')!='X':continue
                    cat=e.get('cat','');name=e.get('name','');span=(float(e.get('ts',0)),float(e.get('ts',0))+float(e.get('dur',0)))
                    if cat=='kernel':kernels.append(span);names[name]+=float(e.get('dur',0))
                    if cat=='gpu_memcpy':copies.append(span)
                def union(spans):
                    total=0.;end=-float('inf')
                    for left,right in sorted(spans):total+=max(0.,right-max(left,end));end=max(end,right)
                    return total
                overlap=union(kernels)+union(copies)-union(kernels+copies)
                categories=Counter()
                for name,duration in names.items():
                    n=name.lower()
                    category=('gemm' if any(x in n for x in ['gemm','matmul','marlin','mm_kernel','cutlass']) else
                        'attention' if any(x in n for x in ['flash','attention','paged']) else
                        'quantization_or_scaling' if any(x in n for x in ['quant','scale','pack','epilogue']) else
                        'nonlinear_norm_rope' if any(x in n for x in ['silu','norm','rotary','softmax']) else 'other')
                    categories[category]+=duration/1000
                write_json(directory/f'{w["id"]}_summary.json',{'gpu_kernel_sum_ms':sum(b-a for a,b in kernels)/1000,
                    'gpu_copy_sum_ms':sum(b-a for a,b in copies)/1000,'kernel_copy_overlap_ms':overlap/1000,
                    'gpu_union_ms':union(kernels+copies)/1000,'categories_ms':categories,
                    'kernel_names_duration_us':names,'trace':str(trace),'output':4,
                    'scope':'Separate profiler run, 1 prefill + 3 decode; not untraced E2E. Name-based categories, other retained. Copy nonoverlap is not by itself critical-path delay.'})
    finally:
        llm.llm_engine.engine_core.shutdown();del llm;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
