#!/usr/bin/env python3
"""Fit strict-heldout monotone raw-kernel-sum to measured E2E calibration."""
from __future__ import annotations
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def pava(y):
    blocks=[]
    for value in y:
        blocks.append([value,1])
        while len(blocks)>1 and blocks[-2][0]/blocks[-2][1]>blocks[-1][0]/blocks[-1][1]:
            right=blocks.pop();blocks[-1][0]+=right[0];blocks[-1][1]+=right[1]
    return [total/count for total,count in blocks for _ in range(int(count))]
def predict(train,x):
    pairs=sorted(train);xs=[a for a,b in pairs];ys=pava([b for a,b in pairs])
    if x<=xs[0]:return ys[0]
    if x>=xs[-1]:return ys[-1]
    for i in range(1,len(xs)):
        if x<=xs[i]:return ys[i-1]+(x-xs[i-1])/(xs[i]-xs[i-1])*(ys[i]-ys[i-1])
def main():
    design=list(csv.DictReader((ROOT/'speed_calibration/design.csv').open()));train={r['policy_id'] for r in design[:7]};rows=[]
    group=ROOT/'speed_calibration_continuous085_w6'
    for r in design:
        values=[json.loads(p.read_text())['elapsed_ms'] for p in sorted((group/'runs'/r['policy_id']).glob('measured_*.json'))]
        if len(values)!=5:raise RuntimeError(f"{r['policy_id']} has {len(values)} samples")
        rows.append({**r,'e2e_median_ms':statistics.median(values)})
    pairs=[(float(r['raw_predicted_linear_ms']),float(r['e2e_median_ms'])) for r in rows if r['policy_id'] in train];dense=next(r for r in rows if r['policy_id']=='p00');scale=float(dense['e2e_median_ms'])/float(dense['raw_predicted_linear_ms'])
    for r in rows:
        r['raw_dense_scaled_ms']=float(r['raw_predicted_linear_ms'])*scale
        r['monotone_prediction_ms']=predict(pairs,float(r['raw_predicted_linear_ms']))
        r['split']='train' if r['policy_id'] in train else 'holdout'
    holdout=[r for r in rows if r['split']=='holdout']
    mae=lambda key:sum(abs(float(r['e2e_median_ms'])-float(r[key])) for r in holdout)/len(holdout)
    output=group
    with (output/'calibration.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metrics={'strict_heldout_points':[r['policy_id'] for r in holdout],'strict_heldout_raw_dense_scaled_mae_ms':mae('raw_dense_scaled_ms'),'strict_heldout_monotone_mae_ms':mae('monotone_prediction_ms'),'monotone_improves_holdout':mae('monotone_prediction_ms')<mae('raw_dense_scaled_ms')}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
