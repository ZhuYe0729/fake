"""Run unchanged paper workflows with experiment-local integer adapters installed."""
from __future__ import annotations

import os
import runpy
import sys

from common import REPO, RUN, WRAPPER, write_json


def main() -> None:
    names = {'speed': 'benchmark_phase_hetero.py', 'prefill_nll': 'evaluate_prefill_nll.py',
             'decode_nll': 'evaluate_prefill_decode_nll.py', 'tasks': 'evaluate_prefill_tasks.py'}
    mode = sys.argv[1]
    script = REPO / 'daso/workflows/paper/runtime' / names[mode]
    os.environ['PHASE_HETERO_TRACE'] = '1' if mode in ['prefill_nll', 'decode_nll'] else '0'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH', 'MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key] = str(WRAPPER)
    from runtime_extensions import install
    phase = install()
    from vllm import LLM
    initialize = LLM.__init__
    host_cache={'prefill_bytes':0,'decode_bytes':0,'pinned_bytes':0}

    def initialize_fixed_cache(instance, *args, **kwargs):
        kwargs.setdefault('kv_cache_memory_bytes', 3 * 1024**3)
        result=initialize(instance, *args, **kwargs)
        import torch
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
        namespace = runpy.run_path(str(script), run_name='paper_workflow')
        # A fixed 3-GiB KV cache leaves room for the original four-request task
        # batches even with sparse workspaces. Keep the workspace cache bounded.
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
        'kv_cache_memory_bytes': 3 * 1024**3,
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
