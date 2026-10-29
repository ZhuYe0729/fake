"""Replay original broad profiling-data validation without touching model caches."""
import csv
import importlib
import json
import time

import numpy as np

from common import RUN,WRAPPER,write_json


def main():
    from modeling.common import kernel_latency_model as generic
    root=WRAPPER/'artifacts/benchmarks/5_kernel_comprehensive';records=[];predictions=[]
    for key in ['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','marlin_nvfp4']:
        module=importlib.import_module('modeling.'+key+'.'+key+'_model')
        if hasattr(module,'CONFIG'):
            raw,skip=generic.load_measurements(module.CONFIG,root)
            data=generic.aggregate_measurements(module.CONFIG,raw)
        elif key=='dense_bf16':data=module.aggregate_measurements(module.load_measurements(root))
        else:
            raw,skip=module.load_measurements(root);data=module.aggregate_measurements(raw)
        for split in ['shape','nk']:
            shapes=data[['m','n','k']].to_numpy();groups=[tuple(x if split=='shape' else x[1:]) for x in shapes]
            unique=sorted(set(groups));rng=np.random.default_rng(144771);rng.shuffle(unique)
            held=set(unique[:max(1,round(len(unique)*.25))]);mask=np.array([x in held for x in groups])
            train=data.loc[~mask].copy();test=data.loc[mask].copy();start=time.perf_counter()
            model=(generic.train_model(module.CONFIG,train) if hasattr(module,'CONFIG') else module.train_model(train))
            fit=time.perf_counter()-start;start=time.perf_counter();estimated=model.predict_frame(test[['m','n','k']]).latency_ms.to_numpy()
            query=time.perf_counter()-start;actual=test.latency_ms.to_numpy();assert np.isfinite(estimated).all()
            errors=abs(estimated-actual)/actual
            records.append({'method':key,'split':split,'seed':144771,'train_shapes':len(train),'test_shapes':len(test),
                'mape_percent':float(errors.mean()*100),'median_ape_percent':float(np.median(errors)*100),
                'p90_ape_percent':float(np.quantile(errors,.9)*100),'mae_ms':float(abs(estimated-actual).mean()),
                'fit_seconds':fit,'query_seconds':query})
            for shape,a,e in zip(test[['m','n','k']].to_dict('records'),actual,estimated,strict=True):
                predictions.append({'method':key,'split':split,**shape,'actual_ms':a,'predicted_ms':e,'ape_percent':abs(e-a)/a*100})
            print(key,split,records[-1]['mape_percent'],flush=True)
    for name,rows in [('broad_latency_validation',records),('broad_latency_predictions',predictions)]:
        with (RUN/f'results/{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(RUN/'results/broad_latency_complete.json',{'passed':True,'metrics':records,
        'scope':'CPU replay of archived original broad kernel-forward measurements; new seed fixed before split. Both exact shapes and N/K groups disjoint. Historical profiling GPU/data, not a new cross-hardware test. All models refit only on training partition; no shared model files overwritten.'})


if __name__=='__main__':main()
