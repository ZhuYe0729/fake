"""All six fixed loads per policy process; real first-token and cycle timings."""
import argparse
import hashlib
import json
import os
import statistics
import time

from common import RUN, write_json
from engine import load, admission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    target = RUN / f'results/measurements/{args.label}.json'
    assert not target.exists()
    design = json.loads((RUN / 'configs/design.json').read_text())
    llm, phase, runtime, initialization = load(args.checkpoint)
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from telemetry import gpu, placement, nvml, HANDLE
    addresses = runtime.addresses()
    first = {'time':None}
    original = llm.llm_engine.output_processor.process_outputs

    def observe(outputs, *a, **kw):
        if first['time'] is None and any(x.new_token_ids for x in outputs):first['time']=time.perf_counter()
        return original(outputs,*a,**kw)

    llm.llm_engine.output_processor.process_outputs = observe
    torch.cuda.reset_peak_memory_stats()
    cases = []
    try:
        for case in design['workloads']:
            check = admission(llm,case)
            path = __import__('pathlib').Path(case['input'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==case['input_sha256']
            tokens = json.loads(path.read_text())
            assert len(tokens)==case['batch'] and all(len(t)==case['prompt'] for t in tokens)
            prompts = [TokensPrompt(prompt_token_ids=t) for t in tokens]
            params = SamplingParams(temperature=0.,max_tokens=case['output'],min_tokens=case['output'],ignore_eos=True,detokenize=False)
            samples = []
            for index in range(design['warmups']+design['iterations']):
                torch.cuda.synchronize()
                before = gpu()
                begin_restore = time.perf_counter()
                if phase.current_phase()=='decode':runtime.restore();runtime.wait_prefill()
                torch.cuda.synchronize()
                restore = (time.perf_counter()-begin_restore)*1000
                first['time'] = None
                begin = time.perf_counter()
                answers = llm.generate(prompts,params,use_tqdm=False)
                torch.cuda.synchronize()
                end = time.perf_counter()
                outputs = [x.outputs[0].token_ids for x in answers]
                assert len(outputs)==case['batch'] and all(len(x)==case['output'] for x in outputs)
                assert first['time'] is not None and first['time']<end
                assert runtime.addresses()==addresses
                samples.append({'iteration':index,'warmup':index<design['warmups'],
                    'e2e_ms':(end-begin)*1000,'ttft_ms':(first['time']-begin)*1000,
                    'tpot_ms':(end-first['time'])*1000/(case['output']-1),
                    'restore_ms':restore,'cycle_ms':restore+(end-begin)*1000,
                    'output_token_ids':outputs,'output_sha256':hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
                    'gpu_before':before,'gpu_after':gpu()})
            medians = {k:statistics.median(s[k] for s in samples if not s['warmup']) for k in ['e2e_ms','ttft_ms','tpot_ms','restore_ms','cycle_ms']}
            cases.append({**case,'admission':check,'samples':samples,'medians':medians})
            print(args.label,case['label'],medians,flush=True)
        audit={}
        for state in ['decode','prefill']:
            if state=='prefill':runtime.restore();runtime.wait_prefill();torch.cuda.synchronize()
            checked=0
            for layer in phase._LAYERS:
                if layer.ph_prefill_method==layer.ph_decode_method:continue
                for name in getattr(layer,'ph_'+state+'_adapter').names:
                    value=getattr(layer,'ph_'+state+'_'+name+('_gpu'if state=='decode'else''))
                    reference=getattr(layer,'ph_'+state+'_cpu')[name].reshape(value.shape)
                    assert torch.equal(value.cpu(),reference),(state,layer.prefix,name)
                    checked+=1
            audit[state]={'exact_equal':True,'tensors':checked}
        write_json(target,{'complete':True,'policy':args.policy,'mode':'arena_graph','placement':placement(),
            'exact_weight_audit':audit,
            'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
            'initialization_ms':initialization,'capture_model_ms':runtime.capture_model_ms,
            'cases':cases,'addresses':addresses,'transfer_bytes_per_generation':runtime.transfer_bytes,
            'memory':{'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'nvml_used_after_bytes':nvml.nvmlDeviceGetMemoryInfo(HANDLE).used,'arena_bytes':runtime.arena_bytes},
            'scope':'One model process per campaign, six loads in fixed order; B1/B8 native decode graphs; per-load warmup, no policy retuning.'})
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':main()
