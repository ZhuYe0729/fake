"""Real gate/up-SiLU-down chains: BF16 interface and measured residual costs."""
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
    design=json.loads((RUN/'configs/design.json').read_text())['mlp_chain']
    raw=torch.load(RUN/'data/profile_weights.pt',weights_only=True)
    scales=torch.load(RUN/'data/activation_scales.pt',weights_only=True)
    states={m:torch.load('/dev/shm/daso_rebuttal_001_20261025/canonical/'+m+'/model.pt',
        mmap=True,weights_only=True)['state_dict'] for m in ['sparse_bf16','sparse_nvfp4']}
    names={k:next(n for n in raw if n.endswith(k)) for k in ['gate_up_proj','down_proj']}
    layers={}
    for method in design['methods']:
        layers[method]={}
        for kind,name in names.items():
            weight=raw[name].cuda()
            if method in states:
                parts=['gate_proj','up_proj'] if kind=='gate_up_proj' else ['down_proj']
                weight=torch.cat([states[method][f'model.layers.0.mlp.{p}.weight'] for p in parts]).cuda()
            layers[method][kind]=(EightBitLinear(method,name,weight,scales) if method in ['sq_int8','fp8'] else compressed_module(method,weight))
    from vllm.model_executor.layers.activation import SiluAndMul
    activation=SiluAndMul();torch.manual_seed(144)

    def timing(call):
        for _ in range(3):out=call()
        torch.cuda.synchronize();samples=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(3):out=call()
            torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000/3)
        return statistics.median(samples),samples

    rows=[]
    for m in design['m']:
        x=torch.randn(m,raw[names['gate_up_proj']].shape[1],device='cuda',dtype=torch.bfloat16)
        for pre in design['methods']:
            gate=layers[pre]['gate_up_proj'];g=gate(x);a=activation(g)
            assert g.dtype==a.dtype==torch.bfloat16 and a.is_contiguous()
            gate_ms,_=timing(lambda:gate(x));act_ms,_=timing(lambda:activation(g))
            for down in design['methods']:
                projection=layers[down]['down_proj']
                separate=projection(a);chain=projection(activation(gate(x)))
                assert separate.dtype==torch.bfloat16 and separate.shape==x.shape
                assert torch.equal(separate,chain),'same deterministic chain and staged inputs must match'
                down_ms,_=timing(lambda:projection(a))
                chain_ms,samples=timing(lambda:projection(activation(gate(x))))
                rows.append({'m':m,'gate_method':pre,'down_method':down,'gate_ms':gate_ms,'silu_ms':act_ms,
                    'down_ms':down_ms,'sum_components_ms':gate_ms+act_ms+down_ms,'chain_ms':chain_ms,
                    'residual_ms':chain_ms-gate_ms-act_ms-down_ms,'bf16_interface':True,
                    'interface_contiguous':a.is_contiguous(),'interface_row_stride':a.stride(0),
                    'interface_width':a.shape[1],'same_staged_chain_output':True,'samples_ms':json.dumps(samples)})
                del separate,chain
            del g,a;gc.collect();torch.cuda.empty_cache()
            print(m,pre,'complete',flush=True)
    with (RUN/'results/mlp_interfaces.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/interfaces_complete.json',{'passed':True,'rows':len(rows),
        'scope':'Actual Qwen3 layer0 BF16 gate/up->SiLU->down; all 49 method pairs. Quantization/scaling within each Linear measured. No compressed inter-layer activation ABI. Chain-minus-components includes cache/launch and synchronization effects, not a pure conversion tax.'})


if __name__=='__main__':main()
