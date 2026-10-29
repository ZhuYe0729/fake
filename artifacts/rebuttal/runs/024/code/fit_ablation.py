"""Refit unchanged calibrators using measured module-level new-action errors."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN.parents[3]/'daso/src'))
import torch
import daso.modeling.quality as quality


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    design=read(RUN/'configs/design.json');rows=[]
    for rid in design['source_runs']:
        source=RUN.parent/rid
        for name,digest in design['sources_sha256'][rid].items():assert sha(source/name)==digest
        assert sha(Path(quality.__file__))==sha(source/'reference/quality_calibrator_source.py')
        frozen=read(source/'configs/frozen_design.json');methods=frozen['methods']
        quality.METHODS=tuple(methods)
        layers=read(source/'configs/model.json')['layers']
        measured={(r['module'],r['phase'],r['method']):float(r['output_rel_mse'])
            for r in csv.DictReader((source/'data/local_errors_new_module.csv').open())}
        for scene in design['scenarios']:
            errors={(r['module'],r['phase'],r['method']):float(r['output_rel_mse'])
                for r in csv.DictReader((source/f'data/local_errors_{scene}.csv').open())}
            assert all(key in errors for key in measured)
            errors.update(measured)
            manifest=read(source/f'configs/calibration_{scene}.json')
            baseline=read(source/f'results/training/{scene}/{manifest[0]["policy_id"]}.json')['avg_nll']
            phases=['prefill'] if scene=='prefill_only' else ['prefill','decode']
            features,labels=[],[]
            for entry in manifest:
                pid=entry['policy_id'];path=source/f'configs/policies/{pid}.json'
                assert sha(path)==entry['policy_sha256']
                result=source/f'results/training/{scene}/{pid}.json'
                if 'source_sha256' in entry:assert sha(result)==entry['source_sha256']
                x=torch.zeros((2,len(methods),4,4),dtype=torch.float64)
                for name,pair in read(path)['method_map'].items():
                    bucket=int(name.split('.')[2])*4//layers
                    kind=quality.FUSED_TYPES.index(name.rsplit('.',1)[-1])
                    for pi,phase in enumerate(phases):
                        method=pair[phase+'_method']
                        if method!='dense_bf16':x[pi,methods.index(method),bucket,kind]+=errors[name,phase,method]
                features.append(x);labels.append(read(result)['avg_nll']-baseline)
            x=torch.stack(features);y=torch.tensor(labels,dtype=torch.float64)
            train=torch.tensor([e['split']=='train' for e in manifest])
            started=time.perf_counter()
            fit=quality.QualityCalibrator(**{k:v for k,v in frozen['quality_fit'].items() if k!='architecture'}).fit(x,y,train)
            fit.save(RUN/f'results/quality_model_{rid}_{scene}.json')
            predicted=fit.predict(x);elapsed=time.perf_counter()-started
            original=read(source/f'results/quality_fit_{scene}.json')['metrics']
            for split,mask in [('train',train),('holdout',~train),
                ('new_methods_holdout',(~train)&torch.tensor([e['origin']==rid for e in manifest]))]:
                error=predicted[mask]-y[mask]
                rows.append({'source_run':rid,'scenario':scene,'split':split,'policies':int(mask.sum()),
                    'original_mae':original[split]['mae'],'module_new_actions_mae':float(error.abs().mean()),
                    'original_rmse':original[split]['rmse'],'module_new_actions_rmse':float(error.square().mean().sqrt()),
                    'fit_seconds':elapsed})
            print(rid,scene,'holdout',rows[-2],flush=True)
    with (RUN/'results/feature_granularity.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(RUN/'results/final_audit.json',{'complete':True,'verified':True,'rows':len(rows),'fits':4,
        'code_sha256':sha(Path(__file__)),'design_sha256':sha(RUN/'configs/design.json'),
        'csv_sha256':sha(RUN/'results/feature_granularity.csv'),
        'scope':'CPU train/old-holdout feature diagnostic only; not independent validation or measured deployment improvement.'})


if __name__=='__main__':main()
