"""Llama-3.1 adaptation of official layer-sequential magnitude/Wanda 2:4 pruning."""
import argparse
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from common import MODEL,RUN,write_json


class CapturedInput(Exception):pass


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['magnitude','wanda'],required=True);args=p.parse_args()
    start=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='eager').to('cuda').eval();loaded=time.perf_counter()-start
    names={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'};records=[];inputs=[];kwargs={}
    if args.method=='wanda':
        blocks=torch.load(RUN/'data/original_sg_128_2048.pt',weights_only=True)[:,:2048]
        def capture(layer,args,kw):inputs.append(args[0].detach().cpu());kwargs.update(kw);raise CapturedInput
        handle=model.model.layers[0].register_forward_pre_hook(capture,with_kwargs=True)
        try:
            for block in blocks:
                try:model(block[None].to('cuda'),use_cache=False)
                except CapturedInput:pass
        finally:handle.remove()
        assert len(inputs)==128
    for index,layer in enumerate(model.model.layers):
        linears=[(n,m) for n,m in layer.named_modules() if n.rsplit('.',1)[-1] in names]
        if args.method=='wanda':
            sums={n:torch.zeros(m.in_features,device='cuda',dtype=torch.float32) for n,m in linears};handles=[]
            for name,module in linears:
                def add(module,values,out,*,name=name):
                    x=values[0].detach().reshape(-1,values[0].shape[-1]).float();sums[name].add_(x.square().sum(0)/128)
                handles.append(module.register_forward_hook(add))
            try:
                for hidden in inputs:layer(hidden.to('cuda'),**kwargs)
            finally:
                for handle in handles:handle.remove()
        for name,module in linears:
            weight=module.weight;metric=weight.abs().float()
            if args.method=='wanda':metric.mul_(sums[name].sqrt()[None])
            grouped=metric.reshape(weight.shape[0],weight.shape[1]//4,4)
            indices=grouped.topk(2,dim=-1,largest=False).indices
            mask=torch.zeros_like(grouped,dtype=torch.bool);mask.scatter_(-1,indices,True)
            weight.masked_fill_(mask.reshape_as(weight),0)
            counts=(weight.reshape(weight.shape[0],-1,4)!=0).sum(-1)
            assert (counts<=2).all() and int(mask.sum())*2==weight.numel()
            records.append({'module':f'model.layers.{index}.{name}','zero_fraction':float((weight==0).float().mean()),
                'mask_zero_fraction':float(mask.float().mean()),'max_nonzero_per_4':int(counts.max())})
            del metric,grouped,indices,mask,counts
        if args.method=='wanda':
            del sums;next_inputs=[]
            for hidden in inputs:
                out=layer(hidden.to('cuda'),**kwargs);next_inputs.append((out[0] if isinstance(out,tuple) else out).cpu())
            inputs=next_inputs
        gc.collect();torch.cuda.empty_cache();print(args.method,index+1,time.perf_counter()-start,flush=True)
    pruning=time.perf_counter()-start-loaded;state={n:t.detach().cpu() for n,t in model.state_dict().items()}
    metadata={'algorithm':args.method,'pattern':'dense_2_4','quantized':False,'model_path':str(MODEL),
        'sample_count':128 if args.method=='wanda' else 0,'modules':records,'load_seconds':loaded,
        'pruning_and_calibration_seconds':pruning,'elapsed_before_save_seconds':time.perf_counter()-start,
        'source':'reference/wanda/provenance.json','adaptation':'Llama-3.1 layer kwargs and vectorized within-group topk; unchanged official score and layer-sequential propagation'}
    target=(RUN/'data/canonical')/args.method/'model.pt';target.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':state,'metadata':metadata},target)
    metadata['total_seconds_including_save']=time.perf_counter()-start;metadata['path']=str(target);metadata['bytes']=target.stat().st_size
    write_json(RUN/f'results/canonical_{args.method}.json',metadata)
    print('saved',target,flush=True)


if __name__=='__main__':main()
