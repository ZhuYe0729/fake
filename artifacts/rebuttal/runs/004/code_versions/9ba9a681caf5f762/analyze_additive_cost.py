"""Whole-model error of frozen Linear costs, with a BF16-only workload offset."""
import csv
import json

import numpy as np

from common import RUN,write_json


def main():
    old=RUN.parent/'003';points=json.loads((old/'configs/evaluation_points.json').read_text())['points']
    design=json.loads((old/'configs/design.json').read_text());workloads={w['id']:w for w in design['workloads']}
    profile={(r['module_type'],r['method'],int(r['m'])):float(r['latency_ms']) for r in csv.DictReader((old/'results/action_profile.csv').open())}
    measured={(r['label'],r['workload']):r for r in csv.DictReader((old/'results/summary_final.csv').open())}
    def linear(pid,w):
        p=json.loads((old/f'configs/policies/{pid}.json').read_text())['method_map'];total=0.
        for name,methods in p.items():
            kind=name.rsplit('.',1)[-1]
            total+=profile[kind,methods['prefill_method'],w['batch']*w['input']]
            if w['output']>1:total+=profile[kind,methods['decode_method'],w['batch']]*(w['output']-1)
        return total
    rows=[]
    for point in points:
        w=workloads[point['workload']];r=measured[point['label'],w['id']]
        baseline=float(measured['uniform_dense_bf16',w['id']]['e2e_ms'])
        fixed=baseline-linear('uniform_dense_bf16',w)
        estimate=linear(point['policy_id'],w);actual=float(r['e2e_ms'])
        rows.append({'label':point['label'],'policy':point['policy_id'],'family':point['family'],'workload':w['id'],
            'scenario':w['scenario'],'actual_ms':actual,'linear_sum_ms':estimate,'bf16_only_fixed_offset_ms':fixed,
            'offset_prediction_ms':estimate+fixed,'raw_ape_percent':abs(estimate-actual)/actual*100,
            'offset_ape_percent':abs(estimate+fixed-actual)/actual*100})
    with (RUN/'results/additive_whole_model.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=[]
    for scenario in ['prefill_only','prefill_decode']:
        values=[r for r in rows if r['scenario']==scenario and r['label']!='uniform_dense_bf16']
        for key in ['raw_ape_percent','offset_ape_percent']:
            a=np.array([r[key] for r in values]);metrics.append({'scenario':scenario,'metric':key,'rows':len(a),
                'mape_percent':float(a.mean()),'median_ape_percent':float(np.median(a)),'p90_ape_percent':float(np.quantile(a,.9))})
    write_json(RUN/'results/additive_cost_complete.json',{'passed':True,'metrics':metrics,
        'scope':'Frozen run003 GPU3 complete-forward profile vs final GPU0 whole-model measurements, same RTX5090 model. Offset fitted only to BF16 per workload, evaluates all other prespecified policies. Includes nonLinear/runtime residual and representation-transfer mismatch; not a strict SLA predictor or a causal split of conversion overhead.'})
    print(json.dumps(metrics))


if __name__=='__main__':main()
