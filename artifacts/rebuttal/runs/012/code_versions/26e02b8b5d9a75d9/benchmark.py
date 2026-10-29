"""Fixed placement, independent process repeats and actual E2E observations."""
import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--policy',required=True);parser.add_argument('--round',type=int,required=True);parser.add_argument('--node',type=int,required=True);args=parser.parse_args()
    run=Path(__file__).resolve().parents[1];design=json.loads((run/'configs/design.json').read_text());spec=design['rounds'][args.round]
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_FLASHINFER_SAMPLER='0',PHASE_HETERO_TRACE='0',
        CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES='4')
    from common import MODEL,WRAPPER,write_json
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH','PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:os.environ[key]=str(WRAPPER)
    from runtime_extensions import install
    phase=install()
    import torch
    from telemetry import gpu,placement
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    llm=LLM(model=str(args.checkpoint),tokenizer=str(MODEL),dtype='bfloat16',enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=False,max_model_len=2120,max_num_seqs=8,
        max_num_batched_tokens=16384,kv_cache_memory_bytes=design['kv_bytes'],gpu_memory_utilization=.9,skip_mm_profiling=True)
    phase.enable_phase_hetero();initial=placement();original_process=llm.llm_engine.output_processor.process_outputs;observe={'first_ms':None,'start':0.}
    settings={name:getattr(phase,name) for name in ['_SYNC_COPY_ENABLED','_RELEASE_PREFILL_ENABLED',
        '_GC_EVERY_APPLY_ENABLED','_GC_DECODE_ENABLED','_WAIT_ONCE_ENABLED']}

    def process_outputs(outputs,*a,**kw):
        if observe['first_ms'] is None and any(o.new_token_ids for o in outputs):observe['first_ms']=(time.perf_counter()-observe['start'])*1000
        return original_process(outputs,*a,**kw)

    llm.llm_engine.output_processor.process_outputs=process_outputs
    ids=torch.randint(100,30000,(8,2048),generator=torch.Generator().manual_seed(spec['seed']))
    prompts=[TokensPrompt(prompt_token_ids=x.tolist()) for x in ids];cases=[]
    try:
        for output in design['outputs']:
            samples=[];params=SamplingParams(temperature=0.,max_tokens=output,min_tokens=output,ignore_eos=True,detokenize=False)
            for i in range(design['warmups']+design['iterations']):
                before=gpu();torch.cuda.synchronize();restore_start=time.perf_counter()
                if phase.current_phase()=='decode':phase.prepare_next_prefill();phase.wait_for_prefill_ready()
                torch.cuda.synchronize();restore=(time.perf_counter()-restore_start)*1000
                observe.update(first_ms=None,start=time.perf_counter())
                answers=llm.generate(prompts,params,use_tqdm=False);torch.cuda.synchronize();elapsed=(time.perf_counter()-observe['start'])*1000
                tokens=[a.outputs[0].token_ids for a in answers];assert len(tokens)==8 and all(len(t)==output for t in tokens)
                assert observe['first_ms'] is not None
                if output>1:assert observe['first_ms']<elapsed*.9,'first-token observer must precede completion'
                samples.append({'iteration':i,'warmup':i<design['warmups'],'e2e_ms':elapsed,'restore_ms':restore,
                    'cycle_ms':restore+elapsed,'ttft_observed_ms':observe['first_ms'],
                    'decode_remainder_ms':elapsed-observe['first_ms'],
                    'output_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),'generated_tokens':8*output,
                    'gpu_before':before,'gpu_after':gpu()})
            assert len({s['output_sha256'] for s in samples})==1,'same fixed prompts/weights must give consistent greedy token IDs'
            measured=[s['e2e_ms'] for s in samples if not s['warmup']]
            cases.append({'scenario':'prefill_only' if output==1 else 'prefill_decode','output':output,
                'median_ms':statistics.median(measured),'samples':samples})
        write_json(run/f'results/speed/node{args.node}/round{args.round}/{args.policy}.json',{'policy':args.policy,
            'round':args.round,'split':spec['split'],'seed':spec['seed'],'node':args.node,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
            'runtime_settings':settings,
            'first_token_observer':'engine_core_before_FINAL_ONLY_filter',
            'placement_after_init':initial,'placement_after_measurement':placement(),'cases':cases,
            'scope':'Original global-phase pipeline, real eager vLLM; restore excluded from E2E and also reported as complete cycle; all methods share lightweight first-output observation.'})
        print(args.node,args.round,args.policy,[(x['scenario'],round(x['median_ms'],2)) for x in cases],flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown();del llm;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
