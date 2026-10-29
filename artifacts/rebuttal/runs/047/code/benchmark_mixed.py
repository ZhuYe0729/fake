"""Actual mixed scheduler routing, paired NLL and closed-loop burst timing."""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

from common import MODEL,RUN,WRAPPER,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--policy',required=True)
    parser.add_argument('--quality',action='store_true');parser.add_argument('--noop',action='store_true');parser.add_argument('--label',required=True)
    args=parser.parse_args();design=json.loads((RUN/'configs/design.json').read_text());attempt=str(time.time_ns())
    assert not args.noop or args.quality and args.policy=='uniform_dense_bf16'
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_FLASHINFER_SAMPLER='0',PHASE_HETERO_TRACE='0',
        PHASE_HETERO_GC_EVERY_APPLY='0',PHASE_HETERO_RELEASE_PREFILL='0',CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES='4')
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key]=str(WRAPPER)
    from runtime_extensions import install
    phase=install()
    import mixed_runtime as mixed
    mixed.hook_inputs()
    import numpy as np
    import torch
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from teacher_force_mixed import TeacherForceBatchProcessor
    from telemetry import gpu,placement,nvml,HANDLE
    start=time.perf_counter()
    llm=LLM(model=args.checkpoint,tokenizer=str(MODEL),dtype='bfloat16',enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=True,max_num_batched_tokens=1024,
        kv_cache_memory_bytes=design['kv_bytes'],gpu_memory_utilization=.90,skip_mm_profiling=True,
        logits_processors=[TeacherForceBatchProcessor]if args.quality else None,**design['engine'])
    initialization=time.perf_counter()-start
    if args.noop:
        mixed.STATE['active']=True;numerical={'passed':True,'scope':'Original BF16 apply; real scheduler classifier and teacher forcing active.'}
    else:
        phase.enable_phase_hetero();mixed.install(phase);numerical=mixed.audit(phase)
    mixed.STATE['track_teacher']=args.quality
    host={};resident={}
    for layer in phase._LAYERS:
        for value in [*layer.ph_prefill_cpu.values(),*layer.ph_decode_cpu.values()]:
            host[value.untyped_storage().data_ptr()]=value.untyped_storage().nbytes()
        for state in ['prefill','decode']:
            if args.noop and state=='decode':continue
            for name in getattr(layer,'ph_'+state+'_adapter').names:
                value=getattr(layer,'ph_'+state+'_'+name+('_gpu'if state=='decode'else''))
                resident[value.untyped_storage().data_ptr()]=value.untyped_storage().nbytes()
    memory={'host_linear_bytes':sum(host.values()),'gpu_linear_bytes':sum(resident.values()),
        'torch_allocated_after_init_bytes':torch.cuda.memory_allocated(),'kv_bytes':design['kv_bytes']}
    cross=sum(l.ph_prefill_method!=l.ph_decode_method for l in phase._LAYERS)

    def generate(prompts,parameters,tag):
        mixed.reset();torch.cuda.synchronize();start=time.perf_counter();seen={};stamps={};final={}
        for i,(prompt,param)in enumerate(zip(prompts,parameters,strict=True)):
            param.output_kind=RequestOutputKind.CUMULATIVE
            llm.llm_engine.add_request(tag+'_'+str(i),prompt,param)
        admission_ms=(time.perf_counter()-start)*1000
        while llm.llm_engine.has_unfinished_requests():
            outputs=llm.llm_engine.step();now=(time.perf_counter()-start)*1000
            for output in outputs:
                if not output.outputs:continue
                key=output.request_id;count=len(output.outputs[0].token_ids);previous=seen.get(key,0)
                if count>previous:stamps.setdefault(key,[]).extend([now]*(count-previous));seen[key]=count
                if output.finished:final[key]=output.outputs[0].token_ids
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        assert len(final)==len(prompts)
        ordered=[final[tag+'_'+str(i)]for i in range(len(prompts))];requests=[]
        for i,tokens in enumerate(ordered):
            times=stamps[tag+'_'+str(i)];assert len(times)==len(tokens)
            requests.append({'request':i,'prompt_tokens':len(prompts[i]['prompt_token_ids']),'output_tokens':len(tokens),
                'ttft_ms':times[0],'completion_ms':times[-1],'tpot_ms':(times[-1]-times[0])/(len(tokens)-1)})
        route=mixed.summary();assert route['mixed_forwards']>0
        assert route['total_prefill_rows']==sum(len(p['prompt_token_ids'])for p in prompts)
        assert route['total_decode_rows']==sum(len(t)-1 for t in ordered)
        if cross:assert route['linear_counts']['mixed']==cross*route['mixed_forwards']
        return ordered,{'e2e_ms':elapsed,'admission_ms':admission_ms,'requests':requests,'scheduler':route,
            'output_tokens_per_second':sum(map(len,ordered))*1000/elapsed,
            'output_sha256':hashlib.sha256(json.dumps(ordered).encode()).hexdigest()}

    try:
        quality_path=Path(design['quality_file'])
        if args.quality:
            assert hashlib.sha256(quality_path.read_bytes()).hexdigest()==design['quality_sha256']
            data=torch.load(quality_path,weights_only=True);assert len(data)==64
        cases=[]
        for case in design['cases']:
            rows=[];batch=len(case['prompts']);torch.cuda.reset_peak_memory_stats()
            if args.quality:
                for offset in range(0,64,batch):
                    tag=args.label+'_'+case['id']+'_'+str(offset);prompts=[];params=[];targets=[];files=[]
                    for i,size in enumerate(case['prompts']):
                        row=data[offset+i];target=row[size:size+case['output']].tolist();assert len(target)==case['output']
                        capture=RUN/f'results/captures/{args.label}/{attempt}/{case["id"]}/{offset+i}.jsonl';assert not capture.exists()
                        prompts.append(TokensPrompt(prompt_token_ids=row[:size].tolist()));targets.append(target);files.append(capture)
                        params.append(SamplingParams(temperature=0.,max_tokens=case['output'],min_tokens=case['output'],ignore_eos=True,
                            extra_args={'teacher_force_target_ids':target,'teacher_force_capture_path':str(capture),'teacher_force_request_key':tag+'_'+str(i)}))
                    tokens,result=generate(prompts,params,tag);assert tokens==targets
                    for i,file in enumerate(files):
                        values=[json.loads(line)for line in file.read_text().splitlines()]
                        assert len(values)==case['output']
                        assert all(v['step']==v['output_length']==j and v['target_id']==targets[i][j] and math.isfinite(v['logprob'])for j,v in enumerate(values))
                        rows.append({'block':offset+i,'avg_nll':-sum(v['logprob']for v in values)/len(values),'token_count':len(values)})
                    print(args.policy,case['id'],len(rows),'/64',flush=True)
                    write_json(RUN/f'results/quality_routes/{args.label}/{attempt}/{case["id"]}/{offset}.json',result['scheduler'])
                cases.append({'case':case,'blocks':rows,'avg_nll':statistics.mean(r['avg_nll']for r in rows)})
            else:
                generator=torch.Generator().manual_seed(design['speed_seed'])
                prompts=[TokensPrompt(prompt_token_ids=torch.randint(100,30000,(size,),generator=generator).tolist())for size in case['prompts']]
                for i in range(design['warmups']+design['iterations']):
                    params=[SamplingParams(temperature=0.,max_tokens=case['output'],min_tokens=case['output'],ignore_eos=True,detokenize=False)for _ in prompts]
                    before=gpu();tokens,result=generate(prompts,params,args.label+'_'+case['id']+'_'+str(i))
                    assert all(len(t)==case['output']for t in tokens)
                    rows.append({'iteration':i,'warmup':i<design['warmups'],'gpu_before':before,'gpu_after':gpu(),
                        'nvml_used_after_bytes':nvml.nvmlDeviceGetMemoryInfo(HANDLE).used,
                        'torch_reserved_after_bytes':torch.cuda.memory_reserved(),**result})
                timed=[r for r in rows if not r['warmup']]
                metrics={k:statistics.median(r[k]for r in timed)for k in ['e2e_ms','output_tokens_per_second']}
                for key in ['ttft_ms','tpot_ms','completion_ms']:
                    values=[r[key]for burst in timed for r in burst['requests']]
                    metrics.update({f'p{q}_{key}':float(np.quantile(values,q/100))for q in [50,95,99]})
                cases.append({'case':case,'iterations':rows,'medians':metrics,
                    'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                print(args.label,case['id'],metrics,flush=True)
        weight_checks={}
        if not args.noop:
            for state in ['prefill','decode']:
                tensors=0;size=0
                for layer in phase._LAYERS:
                    if layer.ph_prefill_method==layer.ph_decode_method:continue
                    for name in getattr(layer,'ph_'+state+'_adapter').names:
                        actual=getattr(layer,'ph_'+state+'_'+name+('_gpu'if state=='decode'else''))
                        expected=getattr(layer,'ph_'+state+'_cpu')[name].reshape(actual.shape)
                        assert torch.equal(actual.cpu(),expected),(state,layer.prefix,name)
                        tensors+=1;size+=actual.numel()*actual.element_size()
                weight_checks[state]={'exact_equal':True,'tensors':tensors,'bytes':size}
        target=RUN/f'results/{"quality"if args.quality else"measurements"}/{args.label}.json'
        write_json(target,{'complete':True,'policy':args.policy,'label':args.label,'noop':args.noop,'quality':args.quality,
            'cases':cases,'routing_audit':numerical,'exact_weight_audit':weight_checks,'memory':memory,'attempt':attempt,
            'initialization_seconds':initialization,'placement':placement(),'physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),
            'design_sha256':hashlib.sha256((RUN/'configs/design.json').read_bytes()).hexdigest()})
    finally:llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
