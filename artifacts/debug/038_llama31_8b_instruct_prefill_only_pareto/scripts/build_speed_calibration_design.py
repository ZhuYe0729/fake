#!/usr/bin/env python3
"""Select fixed, quality-stratified policies for Llama3 prefill E2E calibration."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
KERNEL={'dense_bf16':'dense_bf16','dense_nvfp4':'dense_nvfp4','sparse_bf16':'sparse_bf16','sparse_nvfp4':'sparse_nvfp4','w4a16_ours':'marlin_nvfp4'}
SELECTED=('p00','p01','p02','p03','p04','p37','p39','p42','p45','p52','p60','p68')
def main():
    actions=list(csv.DictReader((ROOT/'action_support.csv').open()))
    lookup={(r['module_name'],r['kernel']):float(r['latency_ms']) for r in actions if r['supported']=='True'}
    nll={r['policy_id']:r for r in csv.DictReader((ROOT/'nll/prefill_only.csv').open())}
    rows=[]
    for policy_id in SELECTED:
        policy=json.loads((ROOT/'policies/prefill_only'/f'{policy_id}.json').read_text())
        raw=sum(lookup[name,KERNEL[item['prefill_method']]] for name,item in policy['method_map'].items())
        counts={method:sum(item['prefill_method']==method for item in policy['method_map'].values()) for method in KERNEL}
        rows.append({'policy_id':policy_id,'policy_json':str(ROOT/'policies/prefill_only'/f'{policy_id}.json'),'raw_predicted_linear_ms':raw,'measured_delta_nll':float(nll[policy_id]['target_delta_nll']),**{f'count_{k}':v for k,v in counts.items()}})
    output=ROOT/'speed_calibration';output.mkdir(exist_ok=True)
    with (output/'design.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'metadata.json').write_text(json.dumps({'scenario':'prefill_only','selection':list(SELECTED),'train_policies':list(SELECTED[:7]),'holdout_policies':list(SELECTED[7:]),'raw_cost':'sum of supported KernelLatencyPredictor M=16384 per-fused-module latencies; conversion/load time excluded from generate-only E2E scope','runner':'five fresh vLLM processes per policy, b=8 input=2048 output=1 eager .9 util'},indent=2)+'\n')
    print(output/'design.csv')
if __name__=='__main__':main()
