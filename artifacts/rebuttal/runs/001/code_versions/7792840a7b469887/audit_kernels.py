from __future__ import annotations

import gc
import os
import platform
import time
import traceback

import torch
import torch.nn.functional as F

from common import RUN, compressed_module, quantize_int8, write_json


def main() -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    results = []
    methods = ['dense_bf16', 'int8', 'int4', 'dense_nvfp4', 'sparse_bf16',
               'sparse_nvfp4', 'w4a16_ours']
    for method in methods:
        started = time.perf_counter()
        row = {'method': method}
        try:
            weight = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16) * .02
            x = torch.randn(32, 512, device='cuda', dtype=torch.bfloat16)
            if method == 'sparse_bf16':
                from cutlass_wrapper import magnitude_prune_2to4_bf16
                weight = magnitude_prune_2to4_bf16(weight)
            elif method == 'sparse_nvfp4':
                from cutlass_wrapper import magnitude_prune_pairwise_4to8_bf16
                weight = magnitude_prune_pairwise_4to8_bf16(weight)
            layer = compressed_module(method, weight)
            try:
                y = layer(x)
            except Exception as exc:
                if method != 'int8':
                    raise
                row['cutlass_error'] = repr(exc)
                layer = compressed_module(method, weight, int8_backend='int_mm')
                row['int8_backend'] = 'int_mm'
                y = layer(x)
            if method == 'int8':
                row.setdefault('int8_backend', 'cutlass')
                wq, ws = quantize_int8(weight)
                xq, xs = quantize_int8(x)
                reference = F.linear(xq.float() * xs, wq.float() * ws)
                error = float((y.float() - reference).square().mean().sqrt())
                row['quantized_reference_rmse'] = error
                if error > .01:
                    raise RuntimeError(f'INT8 quantized reference RMSE {error}')
            elif method == 'int4':
                from cutlass_wrapper import simulate_w4a4_int4_linear
                reference = simulate_w4a4_int4_linear(x, weight)
                error = float((y.float() - reference.float()).square().mean().sqrt())
                row['quantized_reference_rmse'] = error
                if error > .02:
                    raise RuntimeError(f'INT4 quantized reference RMSE {error}')
            torch.cuda.synchronize()
            row.update(ok=True, output_dtype=str(y.dtype),
                       relative_mse=float((y.float() - F.linear(x, weight).float()).square().sum()
                                          / F.linear(x, weight).float().square().sum()),
                       elapsed_seconds=time.perf_counter() - started)
            del layer, x, y, weight
        except Exception as exc:
            row.update(ok=False, error=repr(exc), traceback=traceback.format_exc(),
                       elapsed_seconds=time.perf_counter() - started)
        results.append(row)
        print(row, flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        write_json(RUN / 'results/kernel_audit.json', {
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'python': platform.python_version(),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'results': results})


if __name__ == '__main__':
    main()
