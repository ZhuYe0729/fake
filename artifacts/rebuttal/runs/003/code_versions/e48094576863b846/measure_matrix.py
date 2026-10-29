"""Run one exported policy over its preassigned workloads in a single engine."""
import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

from common import MODEL, REPO, RUN, WRAPPER, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--policy',required=True);parser.add_argument('--mode',choices=['speed','quality','smoke'],required=True)
    parser.add_argument('--workloads',required=True);args=parser.parse_args()
    design=json.loads((RUN/'configs/design.json').read_text())
    selected=set(args.workloads.split(','));workloads=[w for w in design['workloads'] if w['id'] in selected]
    assert selected=={w['id'] for w in workloads}
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    os.environ['PHASE_HETERO_TRACE']='1' if args.mode=='quality' else '0'
    os.environ['CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES']='4'
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key]=str(WRAPPER)
    from runtime_extensions import install
    phase=install()
    import torch
    sys.path.insert(0,str(REPO/'daso/workflows/paper/runtime'))
    from teacher_force_processor import TeacherForceBatchProcessor
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    options=design['engine']
    from gpu_handoff import handoff
    reservation=handoff('release')
    if reservation is not None:assert reservation['reserved_bytes']==0
    llm=LLM(model=str(args.checkpoint),tokenizer=str(MODEL),dtype='bfloat16',
        enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=False,
        max_model_len=options['max_model_len'],max_num_seqs=options['max_num_seqs'],
        max_num_batched_tokens=options['max_num_batched_tokens'],kv_cache_memory_bytes=design['kv_bytes'],
        gpu_memory_utilization=.90,skip_mm_profiling=True,
        logits_processors=[TeacherForceBatchProcessor] if args.mode=='quality' else None)
    phase.enable_phase_hetero()
    def restore():
        if phase.current_phase()=='decode':
            phase.prepare_next_prefill();phase.wait_for_prefill_ready()

    def timed(w,output,tag):
        target=RUN/f'results/speed/{args.policy}/{tag}.json'
        if target.exists():return json.loads(target.read_text())
        generator=torch.Generator().manual_seed(143+w['batch']*1000003+w['input'])
        ids=torch.randint(100,30000,(w['batch'],w['input']),generator=generator)
        prompts=[TokensPrompt(prompt_token_ids=x.tolist()) for x in ids]
        params=SamplingParams(max_tokens=output,min_tokens=output,ignore_eos=True,temperature=0.,detokenize=False)
        rows=[];warmups=1 if args.mode=='smoke' else design['warmups'];iters=1 if args.mode=='smoke' else design['iterations']
        for i in range(warmups+iters):
            restore();torch.cuda.synchronize();start=time.perf_counter()
            answers=llm.generate(prompts,params,use_tqdm=False)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            generated=[len(x.outputs[0].token_ids) for x in answers]
            assert len(answers)==w['batch'] and generated==[output]*w['batch']
            assert all(len(x.prompt_token_ids)==w['input'] for x in answers)
            rows.append({'iteration':i,'warmup':i<warmups,'elapsed_ms':elapsed,
                         'requests':len(answers),'generated_tokens':sum(generated)})
            del answers
        samples=[x['elapsed_ms'] for x in rows if not x['warmup']]
        result={'policy':args.policy,'batch':w['batch'],'input':w['input'],'output':output,
            'median_ms':statistics.median(samples),'mean_ms':statistics.mean(samples),'iterations':rows,
            'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'kv_bytes':design['kv_bytes'],
            'reservation_released':reservation is None or reservation['reserved_bytes']==0}
        if args.mode!='smoke':write_json(target,result)
        return result

    def quality(w):
        key=f'prefill_s{w["input"]}' if w['scenario']=='prefill_only' else w['id']
        target=RUN/f'results/quality/{args.policy}/{key}.json'
        if target.exists():return
        blocks=torch.load(RUN/'data/test_100_4352.pt',weights_only=True)
        captures=RUN/f'results/captures/{args.policy}/{key}/{time.time_ns()}'
        rows=[];phase._TRACE.clear();begin=time.perf_counter()
        if w['scenario']=='prefill_only':
            params=SamplingParams(temperature=0.,max_tokens=1,ignore_eos=True,prompt_logprobs=1)
            for i,block in enumerate(blocks):
                restore();ids=block[:w['input']].tolist()
                answer=llm.generate([TokensPrompt(prompt_token_ids=ids)],params,use_tqdm=False)[0]
                assert answer.prompt_token_ids==ids and len(answer.prompt_logprobs)==len(ids)
                values=[float(answer.prompt_logprobs[j][ids[j]].logprob) for j in range(1,len(ids))]
                rows.append({'block':i,'token_count':len(values),'nll':-sum(values),'avg_nll':-sum(values)/len(values)})
                del answer
        else:
            for start in range(0,len(blocks),w['batch']):
                valid=min(w['batch'],len(blocks)-start)
                chosen=[blocks[start+i] if i<valid else blocks[i-valid] for i in range(w['batch'])]
                prompts=[];params=[];targets=[];files=[]
                for offset,row in enumerate(chosen):
                    ids=row[:w['input']].tolist();forced=row[w['input']:w['input']+w['output']].tolist()
                    capture=captures/f'block_{start+offset:04d}.jsonl'
                    prompts.append(TokensPrompt(prompt_token_ids=ids));targets.append(forced);files.append(capture)
                    params.append(SamplingParams(temperature=0.,max_tokens=w['output'],min_tokens=w['output'],ignore_eos=True,
                        extra_args={'teacher_force_target_ids':forced,'teacher_force_capture_path':str(capture)}))
                restore();answers=llm.generate(prompts,params,use_tqdm=False)
                for i,(answer,forced,file) in enumerate(zip(answers,targets,files,strict=True)):
                    assert answer.outputs[0].token_ids==forced
                    values=[json.loads(line)['logprob'] for line in file.read_text().splitlines()]
                    assert len(values)==w['output']
                    if i<valid:rows.append({'block':start+i,'token_count':len(values),'nll':-sum(values),'avg_nll':-sum(values)/len(values)})
                del answers
        assert len(rows)==100
        count=sum(r['token_count'] for r in rows);loss=sum(r['nll'] for r in rows)
        events={}
        for e in phase._TRACE:events[e['event']]=events.get(e['event'],0)+1
        if w['scenario']=='prefill_decode':assert events.get('enter_decode') and events.get('apply_decode'),events
        write_json(target,{'policy':args.policy,'workload':w,'avg_nll':loss/count,'token_count':count,
            'total_nll':loss,'blocks':rows,'phase_events':events,'elapsed_seconds':time.perf_counter()-begin,
            'prefill_score_batch':1,'decode_score_batch':w['batch'],'last_batch_padding':'repeat samples, excluded from scoring',
            'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'kv_bytes':design['kv_bytes']})
        print('quality',args.policy,key,loss/count,flush=True)

    try:
        for w in workloads:
            if args.mode=='quality':quality(w);continue
            ttft=timed(w,1,f'ttft_b{w["batch"]}_s{w["input"]}')
            main=ttft if w['output']==1 else timed(w,w['output'],w['id'])
            value={'policy':args.policy,'workload':w,'ttft_ms':ttft['median_ms'],'e2e_ms':main['median_ms'],
                'tpot_ms':0. if w['output']==1 else (main['median_ms']-ttft['median_ms'])/(w['output']-1),
                'output_tokens_per_second':w['batch']*w['output']*1000/main['median_ms'],
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'kv_bytes':design['kv_bytes'],'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
            if args.mode!='smoke':write_json(RUN/f'results/speed/{args.policy}/summary_{w["id"]}.json',value)
            print(args.mode,args.policy,w['id'],round(value['e2e_ms'],2),flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown();del llm;gc.collect();torch.cuda.empty_cache()
        handoff('hold',timeout=2,required=False)


if __name__=='__main__':main()
