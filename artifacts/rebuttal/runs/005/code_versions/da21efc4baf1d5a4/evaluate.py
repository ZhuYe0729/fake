"""Separate calibration-speed, validation, and final held-out evaluation."""
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
    p.add_argument('--mode',choices=['calibration_speed','validation','test','test_speed'],required=True);p.add_argument('--scenarios',required=True)
    args=p.parse_args();design=json.loads((RUN/'configs/design.json').read_text());scenarios=args.scenarios.split(',')
    quality=args.mode in ['validation','test']
    if args.mode.startswith('test'):assert (RUN/'configs/selected_policies.json').exists(),'selection must precede any test labels'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    os.environ['PHASE_HETERO_TRACE']='1' if quality else '0'
    os.environ['CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES']='4'
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:os.environ[key]=str(WRAPPER)
    from runtime_extensions import install
    phase=install()
    import torch
    sys.path.insert(0,str(REPO/'daso/workflows/paper/runtime'))
    from teacher_force_processor import TeacherForceBatchProcessor
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    llm=LLM(model=str(args.checkpoint),tokenizer=str(MODEL),dtype='bfloat16',enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=False,max_model_len=2120,max_num_seqs=8,
        max_num_batched_tokens=16384,kv_cache_memory_bytes=design['kv_bytes'],gpu_memory_utilization=.90,
        skip_mm_profiling=True,logits_processors=[TeacherForceBatchProcessor] if quality else None)
    phase.enable_phase_hetero()

    def restore():
        if phase.current_phase()=='decode':phase.prepare_next_prefill();phase.wait_for_prefill_ready()

    try:
        for scenario in scenarios:
            target=RUN/f'results/{args.mode}/{args.policy}/{scenario}.json'
            if target.exists():continue
            start=time.perf_counter();rows=[];output=1 if scenario=='prefill_only' else 64
            if not quality:
                seed=design['speed_test_seed'] if args.mode=='test_speed' else design['speed_calibration_seed']
                ids=torch.randint(100,30000,(8,2048),generator=torch.Generator().manual_seed(seed))
                prompts=[TokensPrompt(prompt_token_ids=x.tolist()) for x in ids]
                params=SamplingParams(temperature=0.,max_tokens=output,min_tokens=output,ignore_eos=True,detokenize=False)
                for i in range(design['warmups']+design['iterations']):
                    restore();torch.cuda.synchronize();before=time.perf_counter()
                    answers=llm.generate(prompts,params,use_tqdm=False);torch.cuda.synchronize()
                    ms=(time.perf_counter()-before)*1000
                    assert len(answers)==8 and all(len(a.outputs[0].token_ids)==output for a in answers)
                    rows.append({'iteration':i,'warmup':i<design['warmups'],'elapsed_ms':ms,
                        'generated_tokens':sum(len(a.outputs[0].token_ids) for a in answers)})
                write_json(target,{'policy':args.policy,'scenario':scenario,'mode':args.mode,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
                    'median_ms':statistics.median(r['elapsed_ms'] for r in rows if not r['warmup']),'iterations':rows,
                    'seed':seed,'batch':8,'input':2048,'output':output,'elapsed_seconds':time.perf_counter()-start})
                continue
            data=torch.load(RUN/f'data/{args.mode}_64_2112.pt',weights_only=True)
            assert len(data)==64
            phase._TRACE.clear();captures=RUN/f'results/captures/{args.mode}/{args.policy}/{scenario}/{time.time_ns()}'
            batch=8
            for offset in range(0,64,batch):
                prompts=[];params=[];files=[];targets=[]
                for i in range(batch):
                    block=data[offset+i];ids=block[:2048].tolist();forced=block[2048:2112].tolist()
                    prompts.append(TokensPrompt(prompt_token_ids=ids));targets.append(forced)
                    capture=captures/f'block_{offset+i:04d}.jsonl';files.append(capture)
                    params.append(SamplingParams(temperature=0.,max_tokens=output,min_tokens=output,ignore_eos=True,
                        prompt_logprobs=1 if scenario=='prefill_only' else None,
                        extra_args=None if scenario=='prefill_only' else {'teacher_force_target_ids':forced,'teacher_force_capture_path':str(capture)}))
                restore();answers=llm.generate(prompts,params,use_tqdm=False)
                for i,a in enumerate(answers):
                    if scenario=='prefill_only':
                        ids=prompts[i]['prompt_token_ids'];values=[float(a.prompt_logprobs[j][ids[j]].logprob) for j in range(1,2048)]
                    else:
                        assert a.outputs[0].token_ids==targets[i]
                        values=[json.loads(line)['logprob'] for line in files[i].read_text().splitlines()]
                        assert len(values)==64
                    rows.append({'block':offset+i,'token_count':len(values),'nll':-sum(values),'avg_nll':-sum(values)/len(values)})
            count=sum(r['token_count'] for r in rows);total=sum(r['nll'] for r in rows);events={}
            for event in phase._TRACE:events[event['event']]=events.get(event['event'],0)+1
            if scenario=='prefill_decode':assert events.get('apply_decode') and events.get('enter_decode')
            write_json(target,{'policy':args.policy,'scenario':scenario,'mode':args.mode,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
                'avg_nll':total/count,'token_count':count,'blocks':rows,'phase_events':events,'quality_score_batch':batch,
                'data_sha256':hashlib.sha256((RUN/f'data/{args.mode}_64_2112.pt').read_bytes()).hexdigest(),
                'elapsed_seconds':time.perf_counter()-start})
            print(args.mode,args.policy,scenario,total/count,flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown();del llm;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
