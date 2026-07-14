#!/usr/bin/env python3
"""Select fixed quality-stratified policies for phase E2E calibration."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
KERNEL={'dense_bf16':'dense_bf16','dense_nvfp4':'dense_nvfp4','sparse_bf16':'sparse_bf16','sparse_nvfp4':'sparse_nvfp4','w4a16_ours':'marlin_nvfp4'}
SELECTED=('p00','p01','p02','p04','p37','p39','p41','p42','p45','p52','p60','p68')
def main():
    actions=list(csv.DictReader((ROOT/'action_support.csv').open()))
    lookup={(r['phase'],r['module_name'],r['kernel']):float(r['latency_ms']) for r in actions if r['supported']=='True'}
    nll={r['policy_id']:r for r in csv.DictReader((ROOT/'nll/prefill_decode.csv').open())}
    rows=[]
    for policy_id in SELECTED:
        legal = ROOT/'speed_calibration_util085/policies'/f'{policy_id}.json'
        policy=json.loads((legal if legal.exists() else ROOT/'policies/prefill_decode'/f'{policy_id}.json').read_text())
        raw_pre=sum(lookup['prefill',name,KERNEL[item['prefill_method']]] for name,item in policy['method_map'].items())
        raw_dec=sum(lookup['decode',name,KERNEL[item['decode_method']]] for name,item in policy['method_map'].items())
        counts={method:sum(item['prefill_method']==method for item in policy['method_map'].values()) for method in KERNEL}
        rows.append({'policy_id':policy_id,'policy_json':str(legal if legal.exists() else ROOT/'policies/prefill_decode'/f'{policy_id}.json'),'raw_predicted_prefill_ms':raw_pre,'raw_predicted_decode_ms':raw_dec,'raw_predicted_linear_ms':raw_pre+80*raw_dec,'measured_delta_nll':float(nll[policy_id]['target_delta_nll']),**{f'count_{k}':v for k,v in counts.items()}})
    output=ROOT/'speed_calibration';output.mkdir(exist_ok=True)
    with (output/'design.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'metadata.json').write_text(json.dumps({'scenario':'prefill_decode','selection':list(SELECTED),'train_policies':list(SELECTED[:7]),'holdout_policies':list(SELECTED[7:]),'raw_cost':'sum supported KernelLatencyPredictor phase latencies: Mpre=32768 + 80*Mdecode=16; conversion excluded until measured E2E','runner':'five fresh continuous phase-heterogeneous vLLM processes per policy, b=16 input=2048 output=80'},indent=2)+'\n')
    print(output/'design.csv')
if __name__=='__main__':main()
