"""Compare frozen Roofline predictions to packaged Qwen Linear measurements."""
import csv
import statistics
import time

from common import RUN,WRAPPER,write_json
from daso.modeling.latency import KernelLatencyBackend
from daso.policy import METHODS


def main():
    started=time.perf_counter()
    backend=KernelLatencyBackend(WRAPPER,RUN/'results/latency_models')
    with (RUN/'results/action_profile.csv').open() as handle:profile=list(csv.DictReader(handle))
    rows=[]
    for value in profile:
        m,n,k=map(int,(value['m'],value['n'],value['k']))
        if m not in [8,16384] or value['method'] not in METHODS:continue
        method=value['method'];kernel='marlin_nvfp4' if method=='w4a16_ours' else method
        queried_m=32 if method=='sparse_nvfp4' and m==8 else m
        prediction=next(item for item in backend.predictor.predict(queried_m,n,k).candidates if item.kernel==kernel)
        assert prediction.supported and prediction.latency_ms is not None
        actual=float(value['latency_ms']);predicted=prediction.latency_ms
        model=backend.predictor.models[kernel]
        seen=any(tuple(map(int,shape))==(queried_m,n,k) for shape in model.train_shapes)
        rows.append({'module_type':value['module_type'],'method':method,
            'phase':'decode' if m==8 else 'prefill','m':m,'queried_m':queried_m,'n':n,'k':k,
            'measured_ms':actual,'predicted_ms':predicted,'absolute_error_ms':abs(actual-predicted),
            'absolute_relative_error':abs(actual-predicted)/actual,
            'prediction_source':prediction.source,'shape_in_original_training_grid':seen})
    with (RUN/'results/latency_prediction_accuracy.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries={}
    for name,group in [('all',rows),('prefill',[r for r in rows if r['phase']=='prefill']),
        ('decode',[r for r in rows if r['phase']=='decode']),
        ('unseen_shapes',[r for r in rows if not r['shape_in_original_training_grid']])]:
        errors=sorted(row['absolute_relative_error'] for row in group)
        summaries[name]={'count':len(group),'mape':statistics.mean(errors),
            'median_ape':statistics.median(errors),'p90_ape':errors[max(0,(9*len(errors)+9)//10-1)]}
    write_json(RUN/'results/latency_prediction_accuracy.json',{'summaries':summaries,
        'elapsed_seconds':time.perf_counter()-started,'scope':'packaged Linear forward; excludes attention, engine and phase-transition overhead',
        'training_overlap':'explicit per-row flag; original exact-grid shapes are not held-out predictions',
        'padding':'sparse FP4 decode queries M=32, compared against actual M=8 padded forward'})
    print(summaries,flush=True)


if __name__=='__main__':main()
