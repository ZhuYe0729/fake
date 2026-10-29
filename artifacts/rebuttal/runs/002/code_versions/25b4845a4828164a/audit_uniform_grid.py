"""Compare uniform anchors before/after the unchanged DP discretization."""
import csv
import json
import math
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def main():
    methods=json.loads((RUN/'configs/extended_design.json').read_text())['methods']
    kinds=['qkv_proj','o_proj','gate_up_proj','down_proj']
    errors={(r['module'],r['phase'],r['method']):float(r['output_rel_mse'])
        for r in csv.DictReader((RUN/'data/local_errors.csv').open())}
    costs={(r['module_type'],'prefill' if int(r['m'])==16384 else 'decode',r['method']):float(r['latency_ms'])
        for r in csv.DictReader((RUN/'results/action_profile.csv').open())}
    names=json.loads((RUN/'configs/policies/uniform_dense_bf16.json').read_text())['method_map']
    rows=[]
    for scenario in ['prefill_only','prefill_decode']:
        fit=json.loads((RUN/f'results/daso8/{scenario}/quality_model.json').read_text())
        predictions=json.loads((RUN/f'results/daso8/{scenario}/quality_predictions.json').read_text())
        manifest=json.loads((RUN/f'configs/calibration_{scenario}.json').read_text())
        phases=['prefill'] if scenario=='prefill_only' else ['prefill','decode']
        quality={};maximum=0.
        for name in names:
            bucket=int(name.split('.')[2])*4//36;kind=name.rsplit('.',1)[-1]
            for pi,phase in enumerate(phases):
                values=[]
                for mi,method in enumerate(methods):
                    local=0. if method=='dense_bf16' else errors[name,phase,method]
                    value=local*fit['coefficients'][pi][mi][bucket][kinds.index(kind)]/fit['feature_scale']
                    quality[name,phase,method]=value;values.append(value)
                maximum+=max(values)
        for mi,method in enumerate(methods):
            values=[sum(quality[name,phase,method] for phase in phases) for name in names]
            raw=fit['bias']+sum(values)
            used=sum(0 if q<=0 else max(1,math.ceil(q*2000/maximum)) for q in values)
            discrete=fit['bias']+used*maximum/2000
            label=manifest[mi if mi<5 else 72+mi-5]['policy_id']
            prediction=next(p for p in predictions if p['policy_id']==label)
            assert abs(raw-prediction['predicted_delta_nll'])<1.e-12
            rows.append({'scenario':scenario,'method':method,'calibration_actual_delta_nll':prediction['actual_delta_nll'],
                'raw_predicted_delta_nll':raw,'discretized_delta_nll':discrete,'quality_quantum':maximum/2000,
                'predicted_linear_ms':sum(costs[name.rsplit('.',1)[-1],phase,method]*(63 if phase=='decode' else 1)
                                          for name in names for phase in phases),
                'bias':fit['bias'],'used_bins':used,'bins':2000})
    (RUN/'results/uniform_grid_audit.json').write_text(json.dumps({'passed':True,
        'scope':'diagnostic only; no refit, new policy, hyperparameter change or test-driven selection',
        'rows':rows},indent=2)+'\n')
    print(json.dumps([r for r in rows if r['method'] in ['sq_int8','fp8']],indent=2))


if __name__=='__main__':main()
