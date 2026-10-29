"""Record real prefill/decode tensor boundaries of frozen exported policies."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
SOURCE=RUN.parent/'017'
sys.path.insert(0,str(SOURCE/'code'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);args=parser.parse_args()
    from common import MODEL,SPEC,WRAPPER,write_json
    design=json.loads((RUN/'configs/design.json').read_text())
    assert hashlib.sha256((SOURCE/f'configs/policies/{args.policy}.json').read_bytes()).hexdigest()==design['policy_sha256'][args.policy]
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_FLASHINFER_SAMPLER='0',PHASE_HETERO_TRACE='0')
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
    checkpoint=Path(SPEC['temp'])/'checkpoints'/args.policy
    llm=LLM(model=str(checkpoint),tokenizer=str(MODEL),dtype='bfloat16',enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=False,max_model_len=256,
        max_num_seqs=8,max_num_batched_tokens=1024,kv_cache_memory_bytes=1024**3,skip_mm_profiling=True)
    runtime.activate()
    rows={};handles=[]
    def metadata(tensor):
        return {'dtype':str(tensor.dtype),'shape':list(tensor.shape),'stride':list(tensor.stride()),'device':str(tensor.device)}
    for layer in phase._LAYERS:
        def hook(module,inputs,output):
            state=phase.current_phase();key=(module.prefix,state)
            if key in rows:return
            x=inputs[0];y=output[0] if isinstance(output,tuple) else output
            assert isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor)
            rows[key]={'module':module.prefix,'phase':state,
                'weight_method':getattr(module,'ph_'+state+'_method'),'input':metadata(x),'output':metadata(y)}
            assert x.dtype==y.dtype==torch.bfloat16,(key,rows[key])
        handles.append(layer.register_forward_hook(hook))
    try:
        ids=json.loads((SOURCE/'configs/speed_inputs.json').read_text())
        outputs=llm.generate([TokensPrompt(prompt_token_ids=x[:128]) for x in ids],
            SamplingParams(temperature=0.,max_tokens=2,min_tokens=2,ignore_eos=True,detokenize=False),use_tqdm=False)
        assert len(outputs)==8 and all(len(x.outputs[0].token_ids)==2 for x in outputs)
        assert len(rows)==256 and {state for _,state in rows}=={'prefill','decode'}
        write_json(RUN/f'results/{args.policy}.json',{'passed':True,'policy':args.policy,'records':list(rows.values()),
            'source_export_config_sha256':hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
            'scope':'Real tensor metadata, no tensor mutation or separate cast inserted; not a latency benchmark.'})
        print(args.policy,'interfaces',len(rows),flush=True)
    finally:
        for handle in handles:handle.remove()
        llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
