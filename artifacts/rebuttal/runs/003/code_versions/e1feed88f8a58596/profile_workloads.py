"""Measure complete forwards for the exact frozen workload shapes on GPU3."""
import csv
import gc
import hashlib
import json
import os
import statistics
import time

import torch

from common import RUN, compressed_module, int8_forward, quantize_int8, write_json
from eightbit import EightBitLinear, activation

PARTS={'qkv_proj':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
       'o_proj':['self_attn.o_proj'],'gate_up_proj':['mlp.gate_proj','mlp.up_proj'],
       'down_proj':['mlp.down_proj']}


@torch.inference_mode()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    design=json.loads((RUN/'configs/design.json').read_text())
    weights=torch.load(RUN/'data/profile_weights.pt',weights_only=True)
    scales=torch.load(RUN/'data/activation_scales.pt',weights_only=True)
    shapes=sorted({w['batch'] for w in design['workloads']}|
                  {w['batch']*w['input'] for w in design['workloads']})
    torch.manual_seed(143)
    audit=[]
    for name,raw in weights.items():
        w,s=quantize_int8(raw.cuda())
        for m in [1,4,8,16,512,2048]:
            x=torch.randn(m,w.shape[1],device='cuda',dtype=torch.bfloat16)
            q,a=quantize_int8(x)
            if m<32:q=torch.nn.functional.pad(q,(0,0,0,32-m))
            ref=(torch._int_mm(q,w.T)[:m].float()*a*s.T).to(x.dtype)
            actual=int8_forward(x,w,s,backend='int_mm')
            fused_q,fused_scale=activation(x,torch.ones(x.shape[1],device=x.device,dtype=x.dtype),True)
            code_difference=(fused_q[:m].to(torch.int16)-q[:m].to(torch.int16)).abs()
            mse=float((actual.float()-ref.float()).square().mean()/ref.float().square().mean().clamp_min(1e-20))
            assert torch.isfinite(actual).all() and mse<1e-5 and int(code_difference.max())<=1,(name,m,mse)
            audit.append({'module':name,'m':m,'relative_mse_vs_unfused':mse,
                'activation_code_max_difference':int(code_difference.max()),
                'activation_code_fraction_different':float((code_difference>0).float().mean()),
                'activation_scale_max_relative_difference':float(((fused_scale-a).abs()/a).max())})
        del w,s,x,q,a,ref,actual,fused_q,fused_scale,code_difference
        gc.collect();torch.cuda.empty_cache()
    write_json(RUN/'results/int8_fusion_audit.json',{'passed':True,'cases':audit,
        'scope':'same INT8 weights/scales and FP32 token-scale formula; fusion only; no SmoothQuant on ordinary INT8',
        'tolerance':'relative output MSE < 1e-5 (same as run002 kernel audit); activation codes differ by at most one from fused division rounding; all NLL remeasured'})
    states={m:torch.load('/dev/shm/daso_rebuttal_001_20261025/canonical/'+m+'/model.pt',
        map_location='cpu',mmap=True,weights_only=True)['state_dict'] for m in ['sparse_bf16','sparse_nvfp4']}
    rows=[];start=time.perf_counter()
    for name,raw in weights.items():
        kind=name.rsplit('.',1)[-1]
        for method in design['uniform_methods']:
            weight=raw.cuda()
            if method in states:
                weight=torch.cat([states[method][f'model.layers.0.{p}.weight'] for p in PARTS[kind]]).cuda()
            layer=(EightBitLinear(method,name,weight,scales) if method in ['sq_int8','fp8']
                   else compressed_module(method,weight,int8_backend='int_mm'))
            for m in shapes:
                x=torch.randn(m,weight.shape[1],device='cuda',dtype=torch.bfloat16)
                for _ in range(5):out=layer(x)
                torch.cuda.synchronize();samples=[]
                repeats=8 if m<=16 else 3
                for _ in range(5):
                    torch.cuda.synchronize();before=time.perf_counter()
                    for _ in range(repeats):out=layer(x)
                    torch.cuda.synchronize();samples.append((time.perf_counter()-before)*1000/repeats)
                rows.append({'module_type':kind,'method':method,'m':m,'n':weight.shape[0],
                    'k':weight.shape[1],'latency_ms':statistics.median(samples),'samples_ms':json.dumps(samples)})
                del x,out;gc.collect();torch.cuda.empty_cache()
            print(kind,method,'complete',round(time.perf_counter()-start,1),flush=True)
            del layer,weight;gc.collect();torch.cuda.empty_cache()
    path=RUN/'results/action_profile.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/profile_complete.json',{'rows':len(rows),'shapes':shapes,'gpu':3,
        'protocol':'5 warmups; 5 rounds of 8 small/3 large forwards; synchronized wall-time median',
        'scope':'complete forward including input smoothing/quantization/output scaling and Python dispatch',
        'seconds':time.perf_counter()-start,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__=='__main__':main()
