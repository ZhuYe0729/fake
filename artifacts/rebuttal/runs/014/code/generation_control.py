"""Native vLLM full-decode graphs for uniform policies, with eager controls."""
import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from common import MODEL, RUN, WRAPPER, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--mode', choices=['eager', 'eager_frozen', 'graph'], required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    target = RUN / f'results/generation_controls/{args.label}.json'
    assert not target.exists(), 'Preserve existing result; use a new label.'
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',
                      VLLM_USE_FLASHINFER_SAMPLER='0', PHASE_HETERO_TRACE='0')
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key] = str(WRAPPER)
    from runtime_extensions import install
    phase = install()
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from telemetry import gpu
    policy = json.loads((RUN / f'configs/policies/{args.policy}.json').read_text())
    assert all(v['prefill_method'] == v['decode_method'] for v in policy['method_map'].values())
    assert policy['default_prefill_method'] == policy['default_decode_method']
    if args.mode != 'eager':
        # A uniform policy needs no phase transition or weight swapping.
        phase._GC_EVERY_APPLY_ENABLED = False
    compilation = ({'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                    'cudagraph_capture_sizes': [8]} if args.mode == 'graph' else None)
    llm = LLM(model=str(args.checkpoint), tokenizer=str(MODEL), dtype='bfloat16',
              enforce_eager=args.mode != 'graph', compilation_config=compilation,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_model_len=2120, max_num_seqs=8, max_num_batched_tokens=16384,
              kv_cache_memory_bytes=6 * 1024**3, gpu_memory_utilization=.9,
              skip_mm_profiling=True)
    if args.mode == 'eager':
        phase.enable_phase_hetero()
    assert phase._ENABLED == (args.mode == 'eager')
    input_path = RUN / 'configs/speed_inputs.json'
    prompts = [TokensPrompt(prompt_token_ids=x) for x in json.loads(input_path.read_text())]
    params = SamplingParams(temperature=0., max_tokens=64, min_tokens=64,
                            ignore_eos=True, detokenize=False)
    observed = {}
    process = llm.llm_engine.output_processor.process_outputs

    def observe(outputs, *a, **kw):
        if observed['first'] is None and any(o.new_token_ids for o in outputs):
            observed['first'] = time.perf_counter()
        return process(outputs, *a, **kw)

    llm.llm_engine.output_processor.process_outputs = observe

    def restore():
        if phase._ENABLED and phase.current_phase() == 'decode':
            phase.prepare_next_prefill()
            phase.wait_for_prefill_ready()
        torch.cuda.synchronize()

    result = {'policy': args.policy, 'mode': args.mode,
              'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
              'phase_enabled': phase._ENABLED, 'compilation': compilation,
              'scope': 'Uniform-policy diagnostic; fixed B8/S2048/O64, no quantizer or kernel changes.'}
    try:
        rows = []
        for index in range(11):
            restore()
            observed.update(first=None)
            before = gpu()
            start = time.perf_counter()
            answers = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
            tokens = [x.outputs[0].token_ids for x in answers]
            assert len(tokens) == 8 and all(len(x) == 64 for x in tokens)
            rows.append({'iteration': index, 'warmup': index < 3,
                         'ttft_ms': (observed['first'] - start) * 1000,
                         'e2e_ms': (end - start) * 1000,
                         'tpot_ms': (end - observed['first']) * 1000 / 63,
                         'output_token_ids': tokens,
                         'output_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                         'gpu_before': before, 'gpu_after': gpu()})
        result['samples'] = rows
        result['medians'] = {k: statistics.median(x[k] for x in rows[3:])
                             for k in ['ttft_ms', 'e2e_ms', 'tpot_ms']}
        result['timed_output_variants'] = len({x['output_sha256'] for x in rows[3:]})
        result['complete'] = True
        write_json(target, result)
        print(args.label, result['medians'], flush=True)
        if args.profile:
            # Separate short real generation; profiler values never enter medians.
            runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
            model = runner.model
            original_forward = model.forward

            def marked(*a, **kw):
                name = 'ModelDecode' if kw['input_ids'].numel() == 8 else 'ModelPrefill'
                with torch.profiler.record_function(name):
                    return original_forward(*a, **kw)

            model.forward = marked
            restore()
            observed.update(first=None)
            params.max_tokens = params.min_tokens = 8
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
            trace = target.with_name(target.stem + '_trace.json')
            profile.export_chrome_trace(str(trace))
            target.with_suffix('.cpu.txt').write_text(profile.key_averages().table(
                sort_by='self_cpu_time_total', row_limit=50))
            result['trace'] = str(trace)
            write_json(target, result)
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
