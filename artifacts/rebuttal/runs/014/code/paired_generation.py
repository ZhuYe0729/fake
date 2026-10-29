"""ABBA replay / eager intervention inside the same initialized vLLM engine."""
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from common import MODEL, RUN, WRAPPER, write_json


def main():
    policy = sys.argv[1]
    assert not (RUN / f'results/paired/{policy}.json').exists(), 'Preserve existing paired result.'
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',
                      VLLM_USE_FLASHINFER_SAMPLER='0', PHASE_HETERO_TRACE='0')
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key] = str(WRAPPER)
    from runtime_extensions import install
    phase = install()
    phase._GC_EVERY_APPLY_ENABLED = False
    import torch
    from telemetry import gpu
    from vllm import LLM, SamplingParams
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm.inputs import TokensPrompt
    checkpoint = (Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_dense_bf16')
                  if policy == 'uniform_dense_bf16' else
                  Path('/dev/shm/daso_rebuttal_014_20261026/checkpoints') / policy)
    llm = LLM(model=str(checkpoint), tokenizer=str(MODEL), dtype='bfloat16',
              enforce_eager=False, compilation_config={'level': 0,
                  'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [8]},
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_model_len=2120, max_num_seqs=8, max_num_batched_tokens=16384,
              kv_cache_memory_bytes=6 * 1024**3, gpu_memory_utilization=.9,
              skip_mm_profiling=True)
    assert not phase._ENABLED
    state = {'mode': 'graph', 'decode_calls': 0}
    original = CUDAGraphWrapper.__call__

    def gated(self, *a, **kw):
        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
            state['decode_calls'] += 1
            if state['mode'] == 'eager':
                return self.runnable(*a, **kw)
        return original(self, *a, **kw)

    CUDAGraphWrapper.__call__ = gated
    observe = {'first': None}
    original_process = llm.llm_engine.output_processor.process_outputs

    def process(outputs, *a, **kw):
        if observe['first'] is None and any(o.new_token_ids for o in outputs):
            observe['first'] = time.perf_counter()
        return original_process(outputs, *a, **kw)

    llm.llm_engine.output_processor.process_outputs = process
    input_path = RUN / 'configs/speed_inputs.json'
    prompts = [TokensPrompt(prompt_token_ids=x) for x in json.loads(input_path.read_text())]
    params = SamplingParams(temperature=0., max_tokens=64, min_tokens=64,
                            ignore_eos=True, detokenize=False)
    order = ['graph'] * 3 + ['eager'] * 3
    for block in range(4):
        order.extend(['graph', 'eager', 'eager', 'graph'] if block % 2 == 0
                     else ['eager', 'graph', 'graph', 'eager'])
    samples = []
    try:
        for index, mode in enumerate(order):
            state.update(mode=mode, decode_calls=0)
            observe['first'] = None
            torch.cuda.synchronize()
            before = gpu()
            start = time.perf_counter()
            answers = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
            tokens = [x.outputs[0].token_ids for x in answers]
            assert len(tokens) == 8 and all(len(x) == 64 for x in tokens)
            assert state['decode_calls'] == 63, state
            samples.append({'iteration': index, 'warmup': index < 6, 'mode': mode,
                'ttft_ms': (observe['first'] - start) * 1000,
                'tpot_ms': (end - observe['first']) * 1000 / 63,
                'e2e_ms': (end - start) * 1000,
                'output_token_ids': tokens,
                'output_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                'decode_calls': state['decode_calls'], 'gpu_before': before, 'gpu_after': gpu()})
        medians = {mode: {metric: statistics.median(x[metric] for x in samples
                    if not x['warmup'] and x['mode'] == mode)
                    for metric in ['ttft_ms', 'tpot_ms', 'e2e_ms']} for mode in ['eager', 'graph']}
        write_json(RUN / f'results/paired/{policy}.json', {'policy': policy, 'order': order,
            'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'samples': samples, 'medians': medians, 'complete': True,
            'scope': 'Same model/engine/weights/metadata construction; only decode wrapper replay vs runnable changes. Uniform frozen policy.'})
        print(policy, medians, flush=True)
    finally:
        CUDAGraphWrapper.__call__ = original
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
