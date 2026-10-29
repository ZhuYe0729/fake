"""Actual 32B module-dimension kernels, explicitly not a full 32B model run."""
import csv
import gc
import json
import os
import statistics
import time

import torch

from common import RUN,compressed_module,write_json
from eightbit import EightBitLinear


@torch.inference_mode()
def main():
    assert (RUN.parent/'003/results/ready_to_seal.json').exists()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    # Verified against the official Qwen3-32B config: q width is heads*head_dim,
    # which is 8192 rather than hidden_size 5120 in this architecture.
    config={'hidden_size':5120,'intermediate_size':25600,'num_attention_heads':64,
        'num_key_value_heads':8,'head_dim':128,'num_hidden_layers':64,
        'source':'https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json'}
    shapes={'qkv_proj':(10240,5120),'o_proj':(5120,8192),
            'gate_up_proj':(51200,5120),'down_proj':(5120,25600)}
    write_json(RUN/'reference/qwen3_32b_dimensions.json',{'config':config,'shapes_n_k':shapes,
        'scope':'Synthetic weights/activations at actual architecture dimensions; no full pretrained 32B model, quality or E2E claim.'})
    methods=['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours','sq_int8','fp8'];rows=[]
    torch.manual_seed(144919)
    for kind,(n,k) in shapes.items():
        weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
        sparse=weight.clone();groups=sparse.reshape(n,k//4,4)
        ids=groups.abs().topk(2,dim=-1,largest=False).indices;groups.scatter_(-1,ids,0);del ids,groups
        for method in methods:
            start=time.perf_counter();w=sparse if method.startswith('sparse_') else weight
            name='model.layers.0.'+('self_attn.' if kind in ['qkv_proj','o_proj'] else 'mlp.')+kind
            layer=(EightBitLinear(method,name,w,{name:torch.ones(k,device='cuda',dtype=torch.bfloat16)}) if method in ['sq_int8','fp8'] else compressed_module(method,w))
            torch.cuda.synchronize();packing=time.perf_counter()-start
            for m in [1,8,2048,16384]:
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                for _ in range(5):out=layer(x)
                torch.cuda.synchronize();assert out.shape==(m,n) and torch.isfinite(out).all()
                samples=[]
                for _ in range(5):
                    start=time.perf_counter()
                    for _ in range(3):out=layer(x)
                    torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000/3)
                rows.append({'architecture':'Qwen3-32B module dimensions','module_type':kind,'method':method,
                    'm':m,'n':n,'k':k,'median_ms':statistics.median(samples),'samples_ms':json.dumps(samples),
                    'pack_seconds':packing,'finite_output':True,'scope':'synthetic layer shape only'})
                del x,out;gc.collect();torch.cuda.empty_cache()
            del layer;gc.collect();torch.cuda.empty_cache();print(kind,method,'complete',flush=True)
        del weight,sparse;gc.collect();torch.cuda.empty_cache()
    with (RUN/'results/large_shapes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/large_shapes_complete.json',{'passed':True,'rows':len(rows),'gpu':6,
        'scope':'112 actual GPU full-forward kernel measurements at genuine 32B N/K shapes, 4 token counts x 7 methods x 4 types. Synthetic per-layer weights; not a pretrained full-model benchmark.'})


if __name__=='__main__':main()
