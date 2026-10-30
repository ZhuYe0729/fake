from __future__ import annotations

import csv
import gc
import hashlib
import json
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from common import MODEL, RUN, compressed_module, write_json
from eightbit import EightBitLinear
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
            for method in ['int8','sq_int8','fp8']:
                modules[kind,method]=(compressed_module(method,weight,int8_backend='int_mm') if method=='int8' else EightBitLinear(method,name,weight,scales)).eval()

            def error(_module,args,_out,kind=kind):
                for phase,x in [('prefill',args[0][:,:2048]),('decode',args[0][:,2048:2112])]:
                    ref=F.linear(x,refs[kind]).float(); mass=float(ref.square().sum())
                    for method in ['int8','sq_int8','fp8']:
                        out=modules[kind,method](x).float();r=accum[phase,kind,method]
                        r[0]+=float((out-ref).square().sum());r[1]+=mass;r[2]+=ref.numel()
            handles.append(layer.get_submodule(hook).register_forward_hook(error))
        for hidden in inputs[:16]:layer(hidden.to('cuda'),**kwargs)
        for h in handles:h.remove()
        for (phase,kind,method),(sse,mass,count) in accum.items():
            rows.append({'module':f'model.layers.{index}.{kind}','layer':index,'phase':phase,
                'method':method,'layer_bucket':index*4//model.config.num_hidden_layers,'fused_type':kind.rsplit('.',1)[-1],
                'sse':sse,'reference_mass':mass,'output_count':count,'output_rel_mse':sse/max(mass,1e-12)})
        inputs=outputs
        del modules,refs,handles,weight,layer
        gc.collect();torch.cuda.empty_cache()
        print('scales and local errors layer',index+1,'seconds',time.perf_counter()-started,flush=True)
    torch.save(scales,RUN/'data/activation_scales.pt')
    torch.save(profile_weights,RUN/'data/profile_weights.pt')
    with (RUN/'data/local_errors_8bit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/smoothing_metadata.json',{
        'blocks':128,'scale_tokens_per_block':2048,'local_error_blocks':16,'alpha':.5,
        'smoothing_scope':['qkv_proj','gate_up_proj'],'elapsed_seconds':time.perf_counter()-started,
        'source':'https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py',
        'formula':'s=clamp(act_max**0.5 / weight_input_channel_max**0.5,1e-5)',
        'activation_scale_sha256':hashlib.sha256((RUN/'data/activation_scales.pt').read_bytes()).hexdigest(),
        'rows':len(rows),'runtime':'candidate-local x/s fused into dynamic per-token activation quantization'})
    del model,inputs,outputs;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
