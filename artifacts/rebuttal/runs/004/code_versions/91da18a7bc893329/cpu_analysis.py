"""Historical preparation costs and strict M-held-out original latency models."""
import csv
import importlib
import json
import time

import numpy as np
import pandas as pd

from common import RUN,write_json


def main():
    rows=[]
    old=RUN.parent/'001';previous=json.loads((old/'results/preparation_costs.json').read_text())
    for r in previous:rows.append({'run':'001',**{k:v for k,v in r.items() if k!='records'}})
    newer=json.loads((RUN.parent/'002/results/preparation_costs.json').read_text())
    rows.append({'run':'002','stage':'SmoothQuant scales and extended local errors',
                 'seconds':newer['activation_scales_and_local_errors']['elapsed_seconds']})
    for scenario in ['prefill_only','prefill_decode']:
        r=newer[scenario]
        rows.append({'run':'002','stage':scenario+' additional labels wall',
            'seconds':r['parallel_wall_seconds'],'allocated_gpus':r['allocated_gpus'],
            'sum_label_elapsed_seconds':r['sum_label_elapsed_seconds'],'new_labels':r['new_labels']})
        for kind in ['daso8','daso5_measured']:
            r=newer[kind+'/'+scenario]
            rows.append({'run':'002','stage':kind+'/'+scenario+' fit+query+DP',
                         'seconds':r['total_seconds'],'policies':r['policies']})
    for file,stage in [('profile_complete.json','360 complete-forward shape measurements'),
                       ('allocation_replay_audit.json','47-policy CPU replay')]:
        r=json.loads((RUN/'reference'/file).read_text())
        rows.append({'run':'003','stage':stage,'source':str(RUN/'reference'/file),
                     'seconds':r.get('seconds',r.get('replay_seconds',r.get('elapsed_seconds')))})
    write_json(RUN/'results/preparation_costs.json',{'records':rows,
        'accounting':'Historical stages overlap in wall time; do not sum all wall times as one elapsed run. Cached stages are inherited, not free from scratch. Validation/test evaluation excluded from preparation.',
        'device':'RTX5090; historical allocations explicitly retained; serial GPU work differs from parallel wall time'})
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (RUN/'results/preparation_costs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    design=json.loads((RUN/'configs/design.json').read_text())
    data=pd.read_csv(RUN/'reference/action_profile.csv');predictions=[];metrics=[]
    for method in ['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','w4a16_ours']:
        key='marlin_nvfp4' if method=='w4a16_ours' else method
        module=importlib.import_module('modeling.'+key+'.'+key+'_model')
        d=data[data.method==method].copy()
        train=d[d.m.isin(design['latency_holdout']['train_m'])].copy()
        test=d[d.m.isin(design['latency_holdout']['test_m'])].copy()
        assert set(map(tuple,train[['m','n','k']].to_numpy())).isdisjoint(set(map(tuple,test[['m','n','k']].to_numpy())))
        start=time.perf_counter()
        if hasattr(module,'CONFIG'):
            from modeling.common.kernel_latency_model import add_physical_columns,train_model
            model=train_model(module.CONFIG,add_physical_columns(module.CONFIG,train))
        else:model=module.train_model(module.add_physical_columns(train))
        fit=time.perf_counter()-start;start=time.perf_counter()
        pred=model.predict_frame(test[['m','n','k']]);query=time.perf_counter()-start
        actual=test.latency_ms.to_numpy();estimated=pred.latency_ms.to_numpy()
        valid=np.isfinite(estimated);ape=np.abs(estimated[valid]-actual[valid])/actual[valid]
        metrics.append({'method':method,'train_rows':len(train),'test_rows':len(test),
            'supported_test_rows':int(valid.sum()),'unsupported_test_rows':int((~valid).sum()),
            'mape_percent':float(ape.mean()*100),'median_ape_percent':float(np.median(ape)*100),
            'p90_ape_percent':float(np.quantile(ape,.9)*100),'max_ape_percent':float(ape.max()*100),
            'fit_seconds':fit,'query_seconds':query,'mae_ms':float(np.abs(estimated[valid]-actual[valid]).mean())})
        for r,estimate in zip(test.to_dict('records'),estimated,strict=True):
            predictions.append({**{k:r[k] for k in ['method','module_type','m','n','k','latency_ms']},
                'predicted_ms':float(estimate) if np.isfinite(estimate) else None,
                'ape_percent':float(abs(estimate-r['latency_ms'])/r['latency_ms']*100) if np.isfinite(estimate) else None,
                'supported':bool(np.isfinite(estimate))})
    for name,records in [('latency_holdout_predictions',predictions),('latency_holdout_metrics',metrics)]:
        with (RUN/f'results/{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)
    write_json(RUN/'results/cpu_analysis_complete.json',{'passed':True,'metrics':metrics,
        'scope':'Original five method-aware model architectures refitted on 4 disjoint M values; 6 held-out M values, same Qwen module N/K; complete-forward wall-time target. No pretrained-model shape leakage; this is a sparse local calibration study, not the original comprehensive predictor test.'})
    print(json.dumps(metrics))


if __name__=='__main__':main()
