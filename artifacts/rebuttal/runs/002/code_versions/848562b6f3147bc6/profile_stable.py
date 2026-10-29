"""Repeated complete-forward timings, with identical host and GPU scope."""
import csv
import gc
import json
import os
import shutil
import statistics
import time

import torch

from common import RUN, compressed_module, write_json
from eightbit import EightBitLinear
from prepare_8bit import SPECS


@torch.inference_mode()
def main():
    weights=torch.load(RUN/'data/profile_weights.pt',weights_only=True)
    scales=torch.load(RUN/'data/activation_scales.pt',weights_only=True)
    states={m:torch.load('/dev/shm/daso_rebuttal_001_20261025/canonical/'+m+'/model.pt',
        map_location='cpu',mmap=True,weights_only=True)['state_dict'] for m in ['sparse_bf16','sparse_nvfp4']}
    rows=[]
    for name,raw in weights.items():
        kind=name.rsplit('.',1)[-1];parts=next(p for _,s,p in SPECS if name.endswith(s))
        for method in ['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours','sq_int8','fp8','sq_fp8']:
            weight=raw.to('cuda')
            if method in states:weight=torch.cat([states[method][f'model.layers.0.{p}.weight'] for p in parts]).to('cuda')
            layer=EightBitLinear(method,name,weight,scales) if method in ['sq_int8','fp8','sq_fp8'] else compressed_module(method,weight)
            for m in [8,16384]:
                x=torch.randn(m,weight.shape[1],dtype=torch.bfloat16,device='cuda')
                for _ in range(10):out=layer(x)
                torch.cuda.synchronize()
                samples=[];wall_samples=[]
                repeats=10 if m==8 else 5
                for _ in range(7):
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();wall=time.perf_counter();a.record()
                    for _ in range(repeats):out=layer(x)
                    b.record();b.synchronize()
                    wall_samples.append((time.perf_counter()-wall)*1000/repeats)
                    samples.append(a.elapsed_time(b)/repeats)
                row={'module_type':kind,'method':method,'m':m,'n':weight.shape[0],'k':weight.shape[1],
                    'supported':True,'latency_ms':statistics.median(wall_samples),'samples_ms':json.dumps(wall_samples),'cuda_event_samples_ms':json.dumps(samples)}
                rows.append(row);print({k:v for k,v in row.items() if k!='samples_ms'},flush=True)
                del x,out;gc.collect();torch.cuda.empty_cache()
            del layer,weight;gc.collect();torch.cuda.empty_cache()
    old=RUN/'results/action_profile.csv';archive=RUN/'results/diagnostics/action_profile_host_launch.csv'
    if not archive.exists():shutil.copy2(old,archive)
    with old.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/action_profile_final.json',{'rows':len(rows),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
        'scope':'complete packaged Linear forward incl Python dispatch, smoothing, activation quantization, real GEMM and scaling',
        'protocol':'10 warmups; 7 rounds of 10 decode / 5 prefill complete forwards; synchronized wall-time median, CUDA events retained',
        'limitation':'phase transfer and model nonlinear layers omitted; whole-model speed measured separately; NVFP4 wrapper disallows CUDA graph capture',
        'previous_host_launch_measurements':'diagnostics/action_profile_host_launch.csv'})


if __name__=='__main__':main()
