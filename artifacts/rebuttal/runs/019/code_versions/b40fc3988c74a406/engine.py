"""One shared arena engine supports the predeclared B1/B8 workload shapes."""
import os
import sys
import time

from common import MODEL, REPO, RUN, WRAPPER


def load(checkpoint, quality=False):
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0',
                      PHASE_HETERO_TRACE='0', CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES='8')
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH', 'MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key] = str(WRAPPER)
    from graph_bindings import install as bindings
    bindings()
    from runtime_extensions import install
    phase = install()
    from arena_runtime import ArenaRuntime
    from vllm import LLM
    import torch
    runtime = ArenaRuntime(phase)
    kwargs = {}
    if quality:
        sys.path.insert(0, str(REPO / 'daso/workflows/paper/runtime'))
        from teacher_force_processor import TeacherForceBatchProcessor
        kwargs['logits_processors'] = [TeacherForceBatchProcessor]
    start = time.perf_counter()
    llm = LLM(model=str(checkpoint), tokenizer=str(MODEL), dtype='bfloat16', enforce_eager=False,
        compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,8]},
        enable_prefix_caching=False, enable_chunked_prefill=False, max_model_len=4168,
        max_num_seqs=8, max_num_batched_tokens=32768, kv_cache_memory_bytes=6*1024**3,
        gpu_memory_utilization=.9, skip_mm_profiling=True, **kwargs)
    runtime.activate()
    torch.cuda.synchronize()
    return llm, phase, runtime, (time.perf_counter()-start)*1000


def admission(llm, case):
    import math
    config = llm.llm_engine.vllm_config.cache_config
    capacity = config.num_gpu_blocks*config.block_size
    required = case['batch']*math.ceil((case['prompt']+case['output'])/config.block_size)*config.block_size
    assert capacity>=required, (case['label'],capacity,required)
    return {'kv_capacity_tokens':capacity,'required_tokens':required,'whole_batch_fits':True}
