from __future__ import annotations

import csv
import gc
import hashlib
import json
import statistics
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from common import MODEL, RUN, compressed_module, write_json
from eightbit import EightBitLinear, smooth_scale, activation
from prepare_sparse import CapturedInput

SPECS = [('self_attn.q_proj','self_attn.qkv_proj',['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj']),
         ('self_attn.o_proj','self_attn.o_proj',['self_attn.o_proj']),
         ('mlp.gate_proj','mlp.gate_up_proj',['mlp.gate_proj','mlp.up_proj']),
         ('mlp.down_proj','mlp.down_proj',['mlp.down_proj'])]


@torch.inference_mode()
def main():
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='eager').to('cuda').eval()
    tokens = torch.load(RUN/'data/calibration_128_2112.pt',weights_only=True)
    inputs, kwargs = [], {}

    def capture(_layer,args,keywords):
        inputs.append(args[0].cpu()); kwargs.update(keywords); raise CapturedInput

    h = model.model.layers[0].register_forward_pre_hook(capture,with_kwargs=True)
    for block in tokens:
        try:model(block[None].to('cuda'),use_cache=False)
        except CapturedInput:pass
    h.remove()
    scales, rows = {}, []
    with (RUN.parent/'001/data/local_errors.csv').open() as f:
        rows.extend(csv.DictReader(f))
    profile_weights = {}
    for index,layer in enumerate(model.model.layers):
        maxima, handles = {}, []
        for hook,kind,parts in SPECS:
            def collect(_module,args,_out,kind=kind):
                value=args[0][:,:2048].abs().reshape(-1,args[0].shape[-1]).amax(0)
                maxima[kind]=value if kind not in maxima else torch.maximum(maxima[kind],value)
            handles.append(layer.get_submodule(hook).register_forward_hook(collect))
        outputs=[]
        for hidden in inputs:
            out=layer(hidden.to('cuda'),**kwargs)
            outputs.append((out[0] if isinstance(out,tuple) else out).cpu())
        for h in handles:h.remove()
        for kind,value in maxima.items():scales[f'model.layers.{index}.{kind}']=value.cpu()
        refs, modules, handles = {}, {}, []
        accum=defaultdict(lambda:[0.,0.,0])
        for hook,kind,parts in SPECS:
            name=f'model.layers.{index}.{kind}'
            weight=torch.cat([layer.get_submodule(part).weight for part in parts])
            refs[kind]=weight
            if index==0:profile_weights[name]=weight.cpu()
            for method in ['sq_int8','fp8','sq_fp8']:
                modules[kind,method]=EightBitLinear(method,name,weight,scales).eval()

            def error(_module,args,_out,kind=kind):
                for phase,x in [('prefill',args[0][:,:2048]),('decode',args[0][:,2048:2112])]:
                    ref=F.linear(x,refs[kind]).float(); mass=float(ref.square().sum())
                    for method in ['sq_int8','fp8','sq_fp8']:
                        out=modules[kind,method](x).float();r=accum[phase,kind,method]
                        r[0]+=float((out-ref).square().sum());r[1]+=mass;r[2]+=ref.numel()
            handles.append(layer.get_submodule(hook).register_forward_hook(error))
        for hidden in inputs[:16]:layer(hidden.to('cuda'),**kwargs)
        for h in handles:h.remove()
        for (phase,kind,method),(sse,mass,count) in accum.items():
            rows.append({'module':f'model.layers.{index}.{kind}','layer':index,'phase':phase,
                'method':method,'layer_bucket':index*4//36,'fused_type':kind.rsplit('.',1)[-1],
                'sse':sse,'reference_mass':mass,'output_count':count,'output_rel_mse':sse/max(mass,1e-12)})
        inputs=outputs
        del modules,refs,handles,weight,layer
        gc.collect();torch.cuda.empty_cache()
        print('scales and local errors layer',index+1,'seconds',time.perf_counter()-started,flush=True)
    torch.save(scales,RUN/'data/activation_scales.pt')
    torch.save(profile_weights,RUN/'data/profile_weights.pt')
    with (RUN/'data/local_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/smoothing_metadata.json',{
        'blocks':128,'scale_tokens_per_block':2048,'local_error_blocks':16,'alpha':.5,
        'smoothing_scope':['qkv_proj','gate_up_proj'],'elapsed_seconds':time.perf_counter()-started,
        'source':'https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py',
        'formula':'s=clamp(act_max**0.5 / weight_input_channel_max**0.5,1e-5)',
        'activation_scale_sha256':hashlib.sha256((RUN/'data/activation_scales.pt').read_bytes()).hexdigest(),
        'rows':len(rows),'runtime':'candidate-local x/s fused into dynamic per-token activation quantization'})
    del model,inputs,outputs;gc.collect();torch.cuda.empty_cache()
    profile(profile_weights,scales)


@torch.inference_mode()
def profile(weights,scales):
    torch.manual_seed(142)
    rows=[]
    states={method:torch.load('/dev/shm/daso_rebuttal_001_20261025/canonical/'+method+'/model.pt',
        map_location='cpu',mmap=True,weights_only=True)['state_dict'] for method in ['sparse_bf16','sparse_nvfp4']}
    for name,raw in weights.items():
        kind=name.rsplit('.',1)[-1]
        parts=next(parts for _,suffix,parts in SPECS if name.endswith(suffix))
        for method in ['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours','sq_int8','fp8','sq_fp8']:
            weight=raw.to('cuda')
            if method in states:weight=torch.cat([states[method][f'model.layers.0.{part}.weight'] for part in parts]).to('cuda')
            layer=EightBitLinear(method,name,weight,scales) if method in ['sq_int8','fp8','sq_fp8'] else compressed_module(method,weight)
            for m in [1,8,2048,16384]:
                x=torch.randn(m,weight.shape[1],dtype=torch.bfloat16,device='cuda')
                for _ in range(5):out=layer(x)
                times=[]
                for _ in range(20):
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();out=layer(x);b.record();b.synchronize();times.append(a.elapsed_time(b))
                assert torch.isfinite(out).all()
                row={'module_type':kind,'method':method,'m':m,'n':weight.shape[0],'k':weight.shape[1],
                    'supported':True,'latency_ms':statistics.median(times),'samples_ms':json.dumps(times)}
                if method in ['sq_int8','fp8','sq_fp8']:
                    q,sa=activation(x,layer.smooth,method=='sq_int8')
                    ref=F.linear(q[:m].float()*sa,layer.weight.float()*layer.scale).to(torch.bfloat16)
                    row['kernel_rel_mse']=float((out.float()-ref.float()).square().sum()/ref.float().square().sum())
                    assert row['kernel_rel_mse']<1.e-5,row
                rows.append(row);print({k:v for k,v in row.items() if k!='samples_ms'},flush=True)
                del x,out
            del layer,weight;gc.collect();torch.cuda.empty_cache()
    with (RUN/'results/action_profile.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])+['kernel_rel_mse']);writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/kernel_audit.json',{'passed':True,'rows':len(rows),
        'scope':'complete Linear forward; inverse smoothing + activation quant + GEMM + output scaling',
        'int8_backend':'torch._int_mm INT8 x INT8 -> INT32 + fused Triton BF16 epilogue',
        'fp8_backend':'vLLM CUTLASS SM120 FP8 E4M3 W8A8 -> BF16',
        'physical_gpu':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')})


if __name__=='__main__':main()
