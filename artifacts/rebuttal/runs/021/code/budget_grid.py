"""Prospective, same-pool budget scan; no new independent test samples."""
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def wait(source,marker):
    while not (source/f'results/{marker}.json').exists():
        write(RUN/'results/worker.json',{'status':'waiting','stage':marker,'source_run':source.name})
        time.sleep(10)


def paired(source,split,pid,scene):
    raw=read(source/f'results/{split}/{pid}/{scene}.json')
    base=read(source/f'results/{split}/uniform_dense_bf16/{scene}.json')
    assert raw['data_sha256']==base['data_sha256']
    delta=[a['avg_nll']-b['avg_nll'] for a,b in zip(raw['blocks'],base['blocks'],strict=True)]
    assert len(delta)==(32 if split=='validation' else 64) and all(math.isfinite(x) for x in delta)
    return statistics.mean(delta),statistics.stdev(delta)/math.sqrt(len(delta))


def select(source,grid):
    candidates=[]
    for scene in ['prefill_only','prefill_decode']:
        for pid,item in read(source/'configs/allocation_pool.json')['policies'].items():
            if scene not in item['selection_scenarios']:continue
            mean,se=paired(source,'validation',pid,scene)
            speed=read(source/f'results/measurements/calibration_{pid}.json')
            case=next(c for c in speed['cases'] if c['output']==(1 if scene=='prefill_only' else 64))
            candidates.append({'policy':pid,'scenario':scene,'validation_delta_nll':mean,'validation_se':se,
                'guarded_delta_nll':0. if pid=='uniform_dense_bf16' else mean+2*se+.005,
                'calibration_e2e_ms':case['medians']['e2e_ms']})
    selected=[]
    for scene in ['prefill_only','prefill_decode']:
        for budget in grid['budgets']:
            feasible=[p for p in candidates if p['scenario']==scene and p['guarded_delta_nll']<=budget]
            winner=min(feasible,key=lambda p:(p['calibration_e2e_ms'],p['policy']))
            selected.append({**winner,'quality_budget':budget,'feasible_candidates':len(feasible)})
    return selected


def main():
    grid=read(RUN/'configs/grid.json')
    assert sha(Path(__file__))==grid['selection_code_sha256']
    selections={}
    for rid,spec in grid['sources'].items():
        source=RUN.parent/rid
        wait(source,'calibration_complete')
        assert sha(source/'configs/allocation_pool.json')==spec['pool_sha256']
        assert sha(source/'configs/frozen_design.json')==spec['design_sha256']
        assert sha(source/'code/select_suite.py')==spec['selector_reference_sha256']
        target=RUN/f'results/selected_{rid}.json'
        if target.exists():selected=read(target)['selected']
        else:
            selected=select(source,grid)
            write(target,{'selected':selected,'selection_code_sha256':grid['selection_code_sha256'],
                'written_at_unix':time.time(),'any_source_confirmation_present':bool(list((source/'results/confirmation').glob('*/*.json'))),
                'scope':'Validation/calibration-only deterministic selection; no confirmation read.'})
        selections[rid]=selected
    rows=[]
    for rid,selected in selections.items():
        source=RUN.parent/rid
        wait(source,'final_audit')
        assert read(source/'results/final_audit.json')['verified']
        original=read(source/'configs/selected_policies.json')['selected']
        for choice in selected:
            if choice['quality_budget'] in [.01,.04]:
                assert choice['policy']==next(x['policy'] for x in original if x['scenario']==choice['scenario'] and x['quality_budget']==choice['quality_budget'])
            mean,se=paired(source,'confirmation',choice['policy'],choice['scenario'])
            rows.append({'source_run':rid,**choice,'confirmation_delta_nll':mean,'confirmation_se':se,
                'mean_budget_violation':mean>choice['quality_budget']})
    with (RUN/'results/budget_points.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(RUN/'results/worker.json',{'status':'complete','stage':'budget_grid_report','points':len(rows),
        'mean_budget_violations':sum(r['mean_budget_violation'] for r in rows)})
    write(RUN/'results/final_audit.json',{'complete':True,'verified':True,
        'scope':'Derived same-pool budget analysis; no new GPU evaluation or independent confirmation samples.',
        'source_manifests_sha256':{rid:sha(RUN.parent/rid/'results/manifest.json') for rid in selections},
        'selection_code_sha256':sha(Path(__file__)),'grid_sha256':sha(RUN/'configs/grid.json'),
        'budget_csv_sha256':sha(RUN/'results/budget_points.csv')})
    print('Completed fixed-pool budget grid',len(rows),'points',flush=True)


if __name__=='__main__':main()
