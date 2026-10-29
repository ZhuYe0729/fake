import csv
import gc
import json
import statistics
import time

import torch

from common import MODEL,RUN,write_json
from int4_marlin import MarlinInt4,pack
from profile_actions import read_tensor


@torch.inference_mode()
def main():
    started=time.perf_counter()
    mapping=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    specs={'self_attn.qkv_proj':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
           'self_attn.o_proj':['self_attn.o_proj'],'mlp.gate_up_proj':['mlp.gate_proj','mlp.up_proj'],
           'mlp.down_proj':['mlp.down_proj']}
    rows=[]
    validation={}
    for kind,parts in specs.items():
        w=torch.cat([read_tensor(f'model.layers.0.{part}.weight',mapping) for part in parts]).to('cuda')
        layer=MarlinInt4(w).eval()
        _,_,reference=pack(w)
        probe=torch.randn(8,w.shape[1],device='cuda',dtype=torch.bfloat16)
        actual=layer(probe).float()
        expected=torch.nn.functional.linear(probe,reference).float()
        relative=float((actual-expected).square().sum()/expected.square().sum())
        if relative>1e-3:raise RuntimeError(f'{kind}: native INT4 reference mismatch {relative}')
        validation[kind]={'quantized_reference_relative_mse':relative,'input_dtype':'bfloat16','output_dtype':'bfloat16'}
        del reference,probe,actual,expected
        for m in [1,8,2048,16384]:
            x=torch.randn(m,w.shape[1],device='cuda',dtype=torch.bfloat16)
            for _ in range(3):layer(x)
            values=[]
            for _ in range(10):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();out=layer(x);end.record();end.synchronize();values.append(begin.elapsed_time(end))
            if not torch.isfinite(out).all():raise RuntimeError('non-finite Marlin INT4 output')
            row={'module_type':kind,'method':'marlin_int4','m':m,'n':w.shape[0],'k':w.shape[1],
                'supported':True,'latency_ms':statistics.median(values),'samples_ms':json.dumps(values)}
            rows.append(row);print(row,flush=True)
            del x,out
        del w,layer;gc.collect();torch.cuda.empty_cache()
    with (RUN/'results/marlin_int4_action_profile.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/marlin_int4_native_validation.json',validation)
    errors=[]
    for name in mapping:
        if not name.startswith('model.layers.') or not name.endswith('_proj.weight'):continue
        w=read_tensor(name,mapping).to('cuda')
        packed,scale,reference=pack(w)
        errors.append({'module':name.removesuffix('.weight'),'method':'marlin_int4','numel':w.numel(),
            'weight_error_squared':float((w.float()-reference.float()).square().sum())})
        del w,packed,scale,reference
    with (RUN/'results/marlin_int4_weight_errors.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    print('Completed optimized INT4 profiling and perturbations',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
