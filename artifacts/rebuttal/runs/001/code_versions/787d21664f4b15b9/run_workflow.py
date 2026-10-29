"""Run unchanged paper workflows with experiment-local integer adapters installed."""
from __future__ import annotations

import os
import faulthandler
import runpy
import sys

from common import REPO, RUN, WRAPPER, write_json


def main() -> None:
    faulthandler.dump_traceback_later(60, repeat=True)
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
    install()
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')
    import torch
    write_json(RUN / f'results/last_memory_{os.getpid()}.json', {
        'mode': mode, 'arguments': sys.argv, 'gpu': torch.cuda.get_device_name(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


if __name__ == '__main__':
    main()
