"""HAWQ weight-error inputs from the exact canonical FP4 quantizer, on CPU."""
import csv
import json
import time

import torch

from common import MODEL, RUN
import cutlass_wrapper.canonical_nvfp4 as canonical
from profile_actions import read_tensor


@torch.inference_mode()
def main():
    # The quantizer is composed of portable tensor operations. Only its public
    # device check is bypassed for this offline statistic, never for inference.
    canonical._check_2d_cuda_float_weight = lambda weight, name='weight': None
    mapping = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    target = RUN / 'results/fp4_weight_quantization_errors.csv'
    started = time.perf_counter()
    with target.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['module','method','numel','weight_error_squared'])
        writer.writeheader()
        count=0
        for name in mapping:
            if not name.startswith('model.layers.') or not name.endswith('_proj.weight'):
                continue
            weight=read_tensor(name,mapping)
            packed=canonical.quantize_nvfp4_canonical_weight(weight)
            dequantized=canonical.dequantize_nvfp4_canonical_weight(packed)
            writer.writerow({'module':name.removesuffix('.weight'),'method':'dense_nvfp4',
                'numel':weight.numel(),
                'weight_error_squared':float((weight.float()-dequantized).square().sum())})
            count+=1
            if count%28==0:print('FP4 errors',count,'seconds',time.perf_counter()-started,flush=True)
    print('Completed',count,'FP4 weight perturbations',flush=True)


if __name__=='__main__':main()
