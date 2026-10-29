"""Run unchanged paper workflows with experiment-local integer adapters installed."""
from __future__ import annotations

import os
import runpy
import sys

from common import REPO, RUN, WRAPPER, SPEC, write_json


def main() -> None:
    names = {'speed': 'benchmark_phase_hetero.py', 'prefill_nll': 'evaluate_prefill_nll.py',
             'decode_nll': 'evaluate_prefill_decode_nll.py', 'tasks': 'evaluate_prefill_tasks.py'}
    mode = sys.argv[1]
    script = REPO / 'daso/workflows/paper/runtime' / names[mode]
    task_protocol = __import__('json').loads((RUN / 'configs/full_task_protocol.json').read_text())
    task_kv_gib = task_protocol['mc_kv_gib'] if mode == 'tasks' else SPEC['task_kv_gib']
    if mode == 'tasks':
        script = RUN / 'code/evaluate_prefill_tasks_batch8.py'
    os.environ['PHASE_HETERO_TRACE'] = '1' if mode in ['prefill_nll', 'decode_nll'] else '0'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH', 'MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key] = str(WRAPPER)
    from runtime_extensions import install
    phase = install()
    from graph_bindings import install as graph_bindings
    graph_bindings()
    from arena_runtime import ArenaRuntime
    runtime = ArenaRuntime(phase)
    from vllm import LLM
    initialize = LLM.__init__
    host_cache={'prefill_bytes':0,'decode_bytes':0,'pinned_bytes':0}

    def initialize_fixed_cache(instance, *args, **kwargs):
        kwargs['kv_cache_memory_bytes'] = task_kv_gib * 1024**3
        result=initialize(instance, *args, **kwargs)
        from admission_check import check
        write_json(RUN / f'results/admission/tasks_{os.getpid()}.json', check(instance, 8, 2048, 1))
        runtime.activate()
        import torch
        if mode == 'tasks':
            import json, math, time, hashlib
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            source = RUN / 'configs/speed_inputs.json'
            token_ids = [ids[:2047] for ids in json.loads(source.read_text())]
            assert len(token_ids) == 8 and all(len(ids) == 2047 for ids in token_ids)
            started = time.perf_counter()
            probes = instance.generate([TokensPrompt(prompt_token_ids=ids) for ids in token_ids],
                SamplingParams(temperature=0., max_tokens=1, min_tokens=1, ignore_eos=True,
                               detokenize=False, prompt_logprobs=1), use_tqdm=False)
            torch.cuda.synchronize()
            assert len(probes) == 8
            for ids, output in zip(token_ids, probes, strict=True):
                assert len(output.prompt_logprobs) == 2047
                assert all(math.isfinite(float(output.prompt_logprobs[j][ids[j]].logprob)) for j in range(1, 2047))
            write_json(RUN / f'results/admission/tasks_max_logprobs_{os.getpid()}.json', {
                'passed':True, 'batch':8, 'prompt_length':2047, 'output_length':1,
                'finite_prompt_logprobs':True, 'input_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'elapsed_seconds':time.perf_counter()-started,
                'process_peak_allocated_through_gate_bytes':torch.cuda.max_memory_allocated(),
                'scope':'Maximum-context MC memory/numeric gate, not a quality benchmark or a timed speed sample.'})
        values={'prefill_bytes':0,'decode_bytes':0,'pinned_bytes':0}
        for layer in phase._LAYERS:
            for name in ['prefill','decode']:
                for tensor in getattr(layer,f'ph_{name}_cpu',{}).values():
                    if not isinstance(tensor,torch.Tensor):continue
                    size=tensor.numel()*tensor.element_size()
                    values[name+'_bytes']+=size
                    if tensor.is_pinned():values['pinned_bytes']+=size
        for key,value in values.items():host_cache[key]=max(host_cache[key],value)
        return result

    LLM.__init__ = initialize_fixed_cache
    if mode in ['prefill_nll', 'tasks']:
        generate = LLM.generate

        def generate_prefill(model, *args, **kwargs):
            if phase._ENABLED and phase.current_phase() == 'decode':
                phase.prepare_next_prefill()
                phase.wait_for_prefill_ready()
            return generate(model, *args, **kwargs)

        LLM.generate = generate_prefill
    sys.argv = [str(script), *sys.argv[2:]]
    if mode == 'tasks':
        import datasets
        datasets.disable_caching()
        from pathlib import Path
        dataset_root=RUN/'data/likelihood_datasets'

        def load_frozen_dataset(path, name=None, *args, split=None, **kwargs):
            namespace=path.replace('/','___').lower()
            roots=[p for p in dataset_root.iterdir() if p.is_dir() and p.name.lower()==namespace]
            if len(roots)!=1:raise ValueError(f'No unique frozen dataset namespace: {path}')
            files=list((roots[0]/(name or 'default')).rglob('*.arrow'))
            auxiliary=RUN/'data/likelihood_auxiliary_datasets'/roots[0].name/(name or 'default')
            files.extend(auxiliary.rglob('*.arrow'))
            tables={}
            for part in ['train','dev','test','validation']:
                matches=[p for p in files if p.stem.endswith('-'+part)]
                if len(matches)>1:raise ValueError(f'Ambiguous frozen split: {path}/{name}/{part}')
                if matches:tables[part]=datasets.Dataset.from_file(str(matches[0]))
            if not tables:raise ValueError(f'No frozen task splits: {path}/{name}')
            result=datasets.DatasetDict(tables)
            return result[split] if split is not None else result

        datasets.load_dataset=load_frozen_dataset
        namespace = runpy.run_path(str(script), run_name='paper_workflow')
        # All policies use MC batch8 with model-specific KV; sparse workspaces stay bounded.
        os.environ['COSPAQ_TASK_SAFE_MODE'] = '0'
        os.environ['CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES'] = '4'
        namespace['main'].__globals__['uses_sparse_bf16'] = lambda entry, base: False
        arguments = namespace['parse_args']()
        if arguments.limit is None:
            pending = tuple(task for task in namespace['TASKS']
                if not (arguments.output_root / arguments.policy / task / 'full/result.json').exists())
            namespace['main'].__globals__['TASKS'] = pending
        namespace['main']()
    else:
        runpy.run_path(str(script), run_name='__main__')
    import torch
    import resource
    memory={
        'mode': mode, 'arguments': sys.argv, 'gpu': torch.cuda.get_device_name(),
        'kv_cache_memory_bytes': task_kv_gib * 1024**3,
        'host_weight_cache_bytes':host_cache,
        'peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    write_json(RUN / f'results/last_memory_{os.getpid()}.json',memory)
    if mode=='speed':
        import json
        from pathlib import Path
        output=Path(sys.argv[sys.argv.index('--output-dir')+1])/'metadata.json'
        metadata=json.loads(output.read_text())
        metadata.update(memory,execution='same_engine_warmup_then_timed',resolved_kv_cache_dtype='bfloat16')
        write_json(output,metadata)


if __name__ == '__main__':
    main()
