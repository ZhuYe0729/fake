"""Actual chunked scheduler burst metrics, routing audit and validation NLL."""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from common import MODEL,REPO,RUN,WRAPPER,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--policy',required=True)
    p.add_argument('--budget',type=int,choices=[1024,4096],required=True);p.add_argument('--noop',action='store_true')
    p.add_argument('--quality',action='store_true');p.add_argument('--full-quality',action='store_true');p.add_argument('--no-chunk',action='store_true')
    p.add_argument('--extension',action='store_true');p.add_argument('--baseline-end',action='store_true');p.add_argument('--repeat',action='store_true');args=p.parse_args()
    scoring=args.quality or args.full_quality
    design=json.loads((RUN/('configs/baseline_extension_design.json' if args.extension else 'configs/design.json')).read_text())
    if args.baseline_end:assert args.extension and args.policy=='uniform_dense_bf16' and not scoring
    serving_root='serving_extended' if args.extension else 'serving'
    if args.repeat:
        assert args.extension and not scoring
        serving_root='serving_repeated'
    full_quality_root='full_quality_extended' if args.extension else 'full_quality'
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_FLASHINFER_SAMPLER='0',PHASE_HETERO_TRACE='0',
        PHASE_HETERO_GC_EVERY_APPLY='0',PHASE_HETERO_RELEASE_PREFILL='0',CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES='4')
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:os.environ[key]=str(WRAPPER)
    from runtime_extensions import install
    phase=install()
    import mixed_runtime as mixed
    mixed.hook_inputs()
    import torch
    import numpy as np
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.inputs import TokensPrompt
    sys.path.insert(0,str(REPO/'daso/workflows/paper/runtime'))
    from teacher_force_mixed import TeacherForceBatchProcessor
    llm=LLM(model=str(args.checkpoint),tokenizer=str(MODEL),dtype='bfloat16',enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=not args.no_chunk,
        max_num_batched_tokens=32768 if args.no_chunk else args.budget,
        kv_cache_memory_bytes=design['kv_bytes'],gpu_memory_utilization=.90,skip_mm_profiling=True,
        logits_processors=[TeacherForceBatchProcessor] if scoring else None,**design['engine'])
    if args.noop:
        assert args.policy=='uniform_dense_bf16';mixed.STATE['active']=True
        numerical={'passed':True,'scope':'Original uniform BF16 apply; row classifier active, no mixed apply installed.'}
    else:
        phase.enable_phase_hetero();mixed.install(phase);numerical=mixed.audit(phase)
    mixed.STATE['track_teacher']=scoring
    write_json(RUN/f'results/routing_audits/{args.policy}_{args.budget}_{"noop" if args.noop else "routed"}{"_repeated" if args.repeat else "_extended" if args.extension else ""}.json',numerical)
    host={};gpu={}
    for layer in phase._LAYERS:
        for tensor in list(layer.ph_prefill_cpu.values())+list(layer.ph_decode_cpu.values()):host[tensor.untyped_storage().data_ptr()]=tensor.untyped_storage().nbytes()
        for name in layer.ph_prefill_adapter.names:
            tensor=getattr(layer,'ph_prefill_'+name);gpu[tensor.untyped_storage().data_ptr()]=tensor.untyped_storage().nbytes()
        if not args.noop:
            for name in layer.ph_decode_adapter.names:
                tensor=getattr(layer,'ph_decode_'+name+'_gpu');gpu[tensor.untyped_storage().data_ptr()]=tensor.untyped_storage().nbytes()
    memory={'host_linear_bytes':sum(host.values()),'gpu_linear_bytes':sum(gpu.values()),
        'allocated_bytes_after_init':torch.cuda.memory_allocated(),'kv_bytes':design['kv_bytes']}
    cross_modules=sum(layer.ph_prefill_method!=layer.ph_decode_method for layer in phase._LAYERS)

    def generate(prompts,parameters,tag):
        mixed.reset();torch.cuda.synchronize();start=time.perf_counter();seen={};times={};final={}
        for i,(prompt,param) in enumerate(zip(prompts,parameters,strict=True)):
            param.output_kind=RequestOutputKind.CUMULATIVE
            llm.llm_engine.add_request(tag+'_'+str(i),prompt,param)
        admitted=(time.perf_counter()-start)*1000
        while llm.llm_engine.has_unfinished_requests():
            outputs=llm.llm_engine.step();now=(time.perf_counter()-start)*1000
            for output in outputs:
                if not output.outputs:continue
                key=output.request_id;count=len(output.outputs[0].token_ids);previous=seen.get(key,0)
                if count>previous:times.setdefault(key,[]).extend([now]*(count-previous));seen[key]=count
                if output.finished:final[key]=output.outputs[0].token_ids
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        assert len(final)==len(prompts)
        ordered=[final[tag+'_'+str(i)] for i in range(len(prompts))]
        per=[]
        for i,tokens in enumerate(ordered):
            stamps=times[tag+'_'+str(i)];assert len(stamps)==len(tokens)
            per.append({'request':i,'prompt_tokens':len(prompts[i]['prompt_token_ids']),'output_tokens':len(tokens),
                'ttft_ms':stamps[0],'completion_ms':stamps[-1],
                'tpot_ms':(stamps[-1]-stamps[0])/(len(tokens)-1) if len(tokens)>1 else 0.})
        scheduler=mixed.summary()
        assert scheduler['total_prefill_rows']==sum(len(p['prompt_token_ids']) for p in prompts),scheduler
        assert scheduler['total_decode_rows']==sum(len(t)-1 for t in ordered),scheduler
        if cross_modules and not args.noop:
            assert scheduler['linear_counts'].get('mixed',0)==cross_modules*scheduler['mixed_forwards'],scheduler
        return ordered,{'e2e_ms':elapsed,'admission_ms':admitted,'requests':per,'scheduler':scheduler,
            'output_tokens_per_second':sum(map(len,ordered))*1000/elapsed,
            'output_sha256':hashlib.sha256(json.dumps(ordered).encode()).hexdigest()}

    try:
        if args.full_quality:
            data=torch.load(RUN/'data/validation_16_4352.pt',weights_only=True)
            for case in design['cases']:
                if case['token_budget']!=args.budget:continue
                if (RUN/f'results/{full_quality_root}/{args.policy}/{case["id"]}.json').exists():continue
                batch=len(case['prompts']);rows=[];schedulers=[]
                captures=RUN/f'results/full_quality_captures/{args.policy}/{case["id"]}/{time.time_ns()}'
                for start in range(0,16,batch):
                    tag='full_'+case['id']+'_'+str(start);prompts=[];params=[];targets=[];files=[]
                    for i,size in enumerate(case['prompts']):
                        row=data[start+i];forced=row[size:size+case['output']].tolist();capture=captures/f'block_{start+i}.jsonl'
                        prompts.append(TokensPrompt(prompt_token_ids=row[:size].tolist()));targets.append(forced);files.append(capture)
                        params.append(SamplingParams(temperature=0.,max_tokens=case['output'],min_tokens=case['output'],ignore_eos=True,
                            extra_args={'teacher_force_target_ids':forced,'teacher_force_capture_path':str(capture),'teacher_force_request_key':tag+'_'+str(i)}))
                    tokens,result=generate(prompts,params,tag);assert tokens==targets
                    assert result['scheduler']['mixed_forwards']>0;schedulers.append(result['scheduler'])
                    for i,file in enumerate(files):
                        values=[json.loads(line) for line in file.read_text().splitlines()]
                        assert len(values)==case['output'] and all(v['step']==v['output_length']==j for j,v in enumerate(values))
                        loss=-sum(v['logprob'] for v in values)
                        rows.append({'block':start+i,'token_count':len(values),'nll':loss,'avg_nll':loss/len(values)})
                write_json(RUN/f'results/{full_quality_root}/{args.policy}/{case["id"]}.json',{
                    'policy':args.policy,'case':case,'blocks':rows,'avg_nll':sum(r['nll'] for r in rows)/sum(r['token_count'] for r in rows),
                    'schedulers':schedulers,'data_sha256':hashlib.sha256((RUN/'data/validation_16_4352.pt').read_bytes()).hexdigest(),
                    'scope':'Actual serving-case B/S/O and genuine mixed scheduler; 16 fixed validation windows; same forced targets across policies.'})
                print('full quality',args.policy,case['id'],'complete',flush=True)
        elif args.quality:
            blocks=torch.load(RUN.parent/'005/data/validation_64_2112.pt',weights_only=True)[:16]
            captures=RUN/f'results/quality_captures/{args.policy}/{"serial" if args.no_chunk else args.budget}/{time.time_ns()}'
            prompts=[];parameters=[];targets=[];files=[]
            for i,row in enumerate(blocks):
                size=512 if i%2==0 else 2048;forced=row[size:size+32].tolist();capture=captures/f'{i}.jsonl'
                prompts.append(TokensPrompt(prompt_token_ids=row[:size].tolist()));targets.append(forced);files.append(capture)
                parameters.append(SamplingParams(temperature=0.,max_tokens=32,min_tokens=32,ignore_eos=True,
                    extra_args={'teacher_force_target_ids':forced,'teacher_force_capture_path':str(capture),
                                'teacher_force_request_key':'quality_'+str(i)}))
            tokens,result=generate(prompts,parameters,'quality');assert tokens==targets
            rows=[]
            for i,file in enumerate(files):
                values=[json.loads(line)['logprob'] for line in file.read_text().splitlines()];assert len(values)==32
                rows.append({'block':i,'token_count':32,'nll':-sum(values),'avg_nll':-sum(values)/32})
            if not args.no_chunk:assert result['scheduler']['mixed_forwards']>0
            write_json(RUN/f'results/quality/{args.policy}/{"nonchunked" if args.no_chunk else args.budget}.json',
                {'blocks':rows,'avg_nll':sum(r['nll'] for r in rows)/(16*32),'scheduler':result['scheduler'],
                 'scope':'16 validation blocks, forced identical target tokens; numerical chunking/phase-routing check, not a downstream accuracy benchmark'})
        else:
            for case in design['cases']:
                if case['token_budget']!=args.budget:continue
                generator=torch.Generator().manual_seed(design['speed_seed']);prompts=[TokensPrompt(prompt_token_ids=torch.randint(100,30000,(s,),generator=generator).tolist()) for s in case['prompts']]
                rows=[];torch.cuda.reset_peak_memory_stats()
                for i in range(design['warmups']+design['iterations']):
                    params=[SamplingParams(temperature=0.,max_tokens=case['output'],min_tokens=case['output'],ignore_eos=True,detokenize=False) for _ in prompts]
                    tokens,r=generate(prompts,params,f'c{i}')
                    assert all(len(t)==case['output'] for t in tokens)
                    assert r['scheduler']['mixed_forwards']>0,'workload must actually exercise mixed tokens'
                    rows.append({'iteration':i,'warmup':i<design['warmups'],**r})
                measured=[r for r in rows if not r['warmup']]
                values={key:statistics.median(r[key] for r in measured) for key in ['e2e_ms','output_tokens_per_second']}
                for key in ['ttft_ms','completion_ms','tpot_ms']:
                    samples=[request[key] for r in measured for request in r['requests']]
                    values['p50_'+key]=float(np.median(samples));values['p95_'+key]=float(np.quantile(samples,.95))
                suffix='_bracket_end' if args.baseline_end else '_noop' if args.noop else ''
                write_json(RUN/f'results/{serving_root}/{args.policy}{suffix}/{case["id"]}.json',
                    {'policy':args.policy,'noop':args.noop,'case':case,'iterations':rows,'memory':memory,
                     'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                     'gpu':os.environ['CUDA_VISIBLE_DEVICES'],**values})
                print(args.policy,case['id'],values,flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown();del llm;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
