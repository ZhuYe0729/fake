"""Uniform measurements for legacy eager, reused-storage eager, and decode graphs."""
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
    parser.add_argument('--mode', choices=['legacy', 'arena_eager', 'arena_graph'], required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--outputs', type=int, nargs='+', default=[1, 64])
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--audit-weights', action='store_true')
    args = parser.parse_args()
    target = RUN / f'results/measurements/{args.label}.json'
    assert not target.exists(), 'Use a new label to preserve existing results.'
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0',
                      PHASE_HETERO_TRACE='0', CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES='8')
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH', 'MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key] = str(WRAPPER)
    from runtime_extensions import install
    phase = install()
    import torch
    from telemetry import gpu, placement, nvml, HANDLE
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    runtime = None
    if args.mode != 'legacy':
        from arena_runtime import ArenaRuntime
        runtime = ArenaRuntime(phase)
    started = time.perf_counter()
    graph = args.mode == 'arena_graph'
    llm = LLM(model=str(args.checkpoint), tokenizer=str(MODEL), dtype='bfloat16',
              enforce_eager=not graph, compilation_config=({'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                  'cudagraph_capture_sizes': [8]} if graph else None),
              enable_prefix_caching=False, enable_chunked_prefill=False, max_model_len=2120,
              max_num_seqs=8, max_num_batched_tokens=16384, kv_cache_memory_bytes=6 * 1024**3,
              gpu_memory_utilization=.9, skip_mm_profiling=True)
    if runtime:
        runtime.activate()
    else:
        phase.enable_phase_hetero()
    torch.cuda.synchronize()
    initialization_ms = (time.perf_counter() - started) * 1000
    addresses = runtime.addresses() if runtime else None
    inputs = RUN / 'configs/speed_inputs.json'
    prompts = [TokensPrompt(prompt_token_ids=x) for x in json.loads(inputs.read_text())]
    observed = {'first': None}
    original_process = llm.llm_engine.output_processor.process_outputs

    def process(outputs, *a, **kw):
        if observed['first'] is None and any(o.new_token_ids for o in outputs):
            observed['first'] = time.perf_counter()
        return original_process(outputs, *a, **kw)

    llm.llm_engine.output_processor.process_outputs = process
    torch.cuda.reset_peak_memory_stats()
    cases = []
    try:
        for output in args.outputs:
            params = SamplingParams(temperature=0., max_tokens=output, min_tokens=output,
                                    ignore_eos=True, detokenize=False)
            rows = []
            for index in range(3 + args.iterations):
                torch.cuda.synchronize()
                before = gpu()
                restore_start = time.perf_counter()
                if phase.current_phase() == 'decode':
                    phase.prepare_next_prefill()
                    phase.wait_for_prefill_ready()
                torch.cuda.synchronize()
                restore_ms = (time.perf_counter() - restore_start) * 1000
                observed['first'] = None
                start = time.perf_counter()
                answers = llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
                end = time.perf_counter()
                tokens = [x.outputs[0].token_ids for x in answers]
                assert len(tokens) == 8 and all(len(x) == output for x in tokens)
                assert observed['first'] is not None
                if output > 1:
                    assert observed['first'] < start + (end - start) * .9
                if runtime:
                    assert runtime.addresses() == addresses, 'Graph weight addresses changed.'
                rows.append({'iteration': index, 'warmup': index < 3,
                    'e2e_ms': (end - start) * 1000, 'ttft_ms': (observed['first'] - start) * 1000,
                    'tpot_ms': (end - observed['first']) * 1000 / (output - 1) if output > 1 else None,
                    'restore_ms': restore_ms, 'cycle_ms': restore_ms + (end - start) * 1000,
                    'output_token_ids': tokens, 'output_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    'gpu_before': before, 'gpu_after': gpu(),
                    'nvml_used_after_bytes': nvml.nvmlDeviceGetMemoryInfo(HANDLE).used})
            timed = [x for x in rows if not x['warmup']]
            metrics = ['e2e_ms', 'ttft_ms', 'restore_ms', 'cycle_ms'] + (['tpot_ms'] if output > 1 else [])
            case = {'output': output, 'samples': rows,
                    'medians': {k: statistics.median(x[k] for x in timed) for k in metrics},
                    'timed_output_variants': len({x['output_sha256'] for x in timed})}
            cases.append(case)
            print(args.label, output, case['medians'], flush=True)
        audit = None
        if runtime and args.audit_weights:
            # Exact representation check outside every timed interval.
            audit = {}
            for state in ['decode', 'prefill']:
                if state == 'prefill':
                    runtime.restore()
                    runtime.wait_prefill()
                    torch.cuda.synchronize()
                checked, checked_bytes = 0, 0
                for layer in phase._LAYERS:
                    if layer.ph_prefill_method == layer.ph_decode_method:
                        continue
                    for name in getattr(layer, 'ph_' + state + '_adapter').names:
                        value = getattr(layer, 'ph_' + state + '_' + name + ('_gpu' if state == 'decode' else ''))
                        reference = getattr(layer, 'ph_' + state + '_cpu')[name].reshape(value.shape)
                        assert torch.equal(value.cpu(), reference), (state, layer.prefix, name)
                        checked += 1
                        checked_bytes += value.numel() * value.element_size()
                audit[state] = {'exact_equal': True, 'tensors': checked, 'bytes': checked_bytes}
        host = {}
        for layer in phase._LAYERS:
            for value in [*layer.ph_prefill_cpu.values(), *layer.ph_decode_cpu.values()]:
                host[value.untyped_storage().data_ptr()] = value.untyped_storage().nbytes()
        write_json(target, {'complete': True, 'policy': args.policy, 'mode': args.mode,
            'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(), 'cases': cases,
            'initialization_ms': initialization_ms, 'placement': placement(),
            'memory': {'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'torch_allocated_after_bytes': torch.cuda.memory_allocated(),
                       'sampled_nvml_max_used_bytes': max(x['nvml_used_after_bytes'] for c in cases for x in c['samples']),
                       'host_weight_bytes': sum(host.values()),
                       'phase_reused_arena_bytes': runtime.arena_bytes if runtime else None},
            'transfer_bytes_per_generation': runtime.transfer_bytes if runtime else None,
            'addresses': addresses, 'exact_weight_audit': audit, 'sparse_cache_env_requested': 8,
            'scope': 'Stable per-layer reused storage; async H2D retained, restore separately reported. Native decode-only graphs, no quantizer/kernel changes.'})
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
