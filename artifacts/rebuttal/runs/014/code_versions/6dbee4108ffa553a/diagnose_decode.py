"""Unprofiled generation, fixed-KV forward replay, and separate CPU/CUDA traces."""
import argparse
import copy
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
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
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
    from vllm.forward_context import get_forward_context, override_forward_context
    from vllm.inputs import TokensPrompt
    from telemetry import gpu, placement

    result = {'policy': args.policy, 'label': args.label,
              'placement': placement(), 'gpu_initial': gpu(),
              'scope': 'Diagnosis only; does not replace run013 Table 2.'}
    target = RUN / f'results/{args.label}/summary.json'
    llm = LLM(model=str(args.checkpoint), tokenizer=str(MODEL), dtype='bfloat16',
              enforce_eager=True, enable_prefix_caching=False,
              enable_chunked_prefill=False, max_model_len=2120,
              max_num_seqs=8, max_num_batched_tokens=16384,
              kv_cache_memory_bytes=6 * 1024**3, gpu_memory_utilization=.9,
              skip_mm_profiling=True)
    phase.enable_phase_hetero()
    runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    model = runner.model
    input_path = RUN / 'configs/speed_inputs.json'
    result['input_sha256'] = hashlib.sha256(input_path.read_bytes()).hexdigest()
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
        if phase.current_phase() == 'decode':
            phase.prepare_next_prefill()
            phase.wait_for_prefill_ready()
        torch.cuda.synchronize()

    captured = {}

    def save_decode(module, a, kw):
        if kw['input_ids'].numel() == 8:
            captured.update(kwargs=kw, context=get_forward_context())

    try:
        rows = []
        with torch.inference_mode():
            for index in range(8):
                restore()
                observed.update(first=None)
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
                             'output_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                             'gpu_after': gpu()})
            result['generation'] = rows
            result['generation_medians'] = {k: statistics.median(x[k] for x in rows[3:])
                                            for k in ['ttft_ms', 'e2e_ms', 'tpot_ms']}
            write_json(target, result)
            print(args.label, 'generation', result['generation_medians'], flush=True)

            hook = model.register_forward_pre_hook(save_decode, with_kwargs=True)
            restore()
            observed.update(first=None)
            llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            hook.remove()
            kwargs = {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                      for k, v in captured['kwargs'].items()}
            context = copy.copy(captured['context'])
            context.attn_metadata = copy.deepcopy(context.attn_metadata)
            result['fixed_inputs'] = {k: v.tolist() for k, v in kwargs.items()
                                      if isinstance(v, torch.Tensor)}
            metadata = next(iter(context.attn_metadata.values()))
            result['metadata_type'] = type(metadata).__name__
            result['metadata_fields'] = {k: v.tolist() if isinstance(v, torch.Tensor) else v
                                         for k, v in vars(metadata).items()
                                         if k in ['seq_lens', 'max_seq_len', 'max_query_len',
                                                  'query_start_loc', 'num_actual_tokens']}

            def forward():
                with override_forward_context(context):
                    return model.compute_logits(model(**kwargs))

            def measure(call, count=30):
                for _ in range(5):
                    call()
                torch.cuda.synchronize()
                samples = []
                for _ in range(5):
                    begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    start = time.perf_counter()
                    for _ in range(count):
                        call()
                    finish.record()
                    finish.synchronize()
                    samples.append({'wall_ms': (time.perf_counter() - start) * 1000 / count,
                                    'stream_elapsed_ms': begin.elapsed_time(finish) / count})
                return {'samples': samples, 'calls_per_sample': count,
                        **{k: statistics.median(x[k] for x in samples) for k in samples[0]}}

            reference = forward().clone()
            result['fixed_eager'] = measure(forward)
            actual = forward()
            result['fixed_eager_max_abs_diff'] = float((actual.float() - reference.float()).abs().max())
            write_json(target, result)
            print(args.label, 'fixed eager', result['fixed_eager'], flush=True)

            original_apply = phase.PhaseHeteroMyTestLinearMethod.apply

            def marked(self, layer, x, bias=None):
                with torch.profiler.record_function('Linear/' + layer.prefix):
                    return original_apply(self, layer, x, bias)

            phase.PhaseHeteroMyTestLinearMethod.apply = marked
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                for _ in range(3):
                    with torch.profiler.record_function('FixedDecode'):
                        forward()
                torch.cuda.synchronize()
            phase.PhaseHeteroMyTestLinearMethod.apply = original_apply
            trace = target.parent / 'fixed_eager_trace.json'
            profile.export_chrome_trace(str(trace))
            (target.parent / 'fixed_eager_cpu.txt').write_text(profile.key_averages().table(
                sort_by='self_cpu_time_total', row_limit=45))
            result['fixed_trace'] = str(trace)

            # This bypasses only the generic phase wrapper, keeping its exact kernel adapters.
            def direct(self, layer, x, bias=None):
                return self.decode_apply(layer, x, bias)

            phase.PhaseHeteroMyTestLinearMethod.apply = direct
            result['fixed_direct_dispatch'] = measure(forward)
            result['direct_max_abs_diff'] = float((forward().float() - reference.float()).abs().max())
            phase.PhaseHeteroMyTestLinearMethod.apply = original_apply
            write_json(target, result)

            # Static fixed-KV forward replay only: scheduler/sampling are excluded.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_logits = forward()
            graph.replay()
            torch.cuda.synchronize()
            result['graph_max_abs_diff'] = float((graph_logits.float() - reference.float()).abs().max())
            result['fixed_graph'] = measure(graph.replay)
            result['gpu_final'] = gpu()
            result['complete'] = True
            write_json(target, result)
            print(args.label, 'complete', {k: result[k] for k in [
                'fixed_direct_dispatch', 'fixed_graph', 'graph_max_abs_diff']}, flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
