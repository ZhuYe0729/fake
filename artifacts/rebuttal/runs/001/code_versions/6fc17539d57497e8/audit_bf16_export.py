"""Check every BF16 exported tensor against the unmodified original checkpoint."""
import json
import time

import torch
from safetensors import safe_open

from baseline_worker import TEMP
from common import MODEL,RUN,write_json


def main():
    started=time.perf_counter()
    mapping=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    rows=[];covered=set()
    with safe_open(TEMP/'checkpoints/uniform_dense_bf16/model.safetensors',framework='pt',device='cpu') as exported:
        for name in exported.keys():
            source=name.replace('.ph_prefill_weight','.weight')
            if '.qkv_proj.' in source:
                parts=[source.replace('.qkv_proj.',f'.{part}_proj.') for part in ['q','k','v']]
            elif '.gate_up_proj.' in source:
                parts=[source.replace('.gate_up_proj.',f'.{part}_proj.') for part in ['gate','up']]
            else:parts=[source]
            values=[]
            for part in parts:
                with safe_open(MODEL/mapping[part],framework='pt',device='cpu') as original:
                    values.append(original.get_tensor(part))
            expected=torch.cat(values) if len(values)>1 else values[0]
            actual=exported.get_tensor(name)
            equal=torch.equal(actual,expected)
            rows.append({'exported':name,'original':parts,'numel':actual.numel(),'bitwise_equal':equal})
            covered.update(parts)
            if not equal:raise ValueError(name)
    assert covered==set(mapping),(set(mapping)-covered,covered-set(mapping))
    write_json(RUN/'results/bf16_export_audit.json',{'all_tensors_bitwise_equal':True,
        'original_tensors':len(covered),'exported_tensors':len(rows),
        'elapsed_seconds':time.perf_counter()-started,'records':rows})
    print('All',len(covered),'original tensors exactly preserved across',len(rows),'exports',flush=True)


if __name__=='__main__':main()
