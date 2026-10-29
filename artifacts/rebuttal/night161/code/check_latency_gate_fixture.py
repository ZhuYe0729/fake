"""CPU-only synthetic end-to-end audit; never writes real reviewer deliveries."""
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

real=Path(__file__).resolve().parents[1];root=real/f'protocol_check/latency_gate_synthetic_{time.time_ns()}'
night=root/'night161';run=root/'runs/041';out=night/'latency_validation'
for p in [run/'configs',run/'results/measurements',out]:p.mkdir(parents=True,exist_ok=True)
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
design=json.loads((real.parent/'runs/041/configs/design.json').read_text());write(run/'configs/design.json',design)
write(run/'results/final_audit.json',{'verified':True,'SYNTHETIC_ONLY':True})
with(run/'results/matched_points.csv').open('w')as handle:
    writer=csv.DictWriter(handle,fieldnames=['policy','delta_nll']);writer.writeheader()
    writer.writerows({'policy':p['policy_id'],'delta_nll':.01}for p in design['points'])
for ri,order in enumerate(design['rounds']):
    for pid in order:
        point=next((p for p in design['points']if p['policy_id']==pid),None)
        cases=[]
        for output in ([1,64]if point is None else[1 if point['scenario']=='prefill_only'else 64]):
            bf=1000. if output==1 else 2000.
            value=bf if point is None else bf*point['latency_fraction']*(.9 if point['allocator']=='daso'else.92)
            restore=400. if point is not None and point['phase_mode']=='phase_pair'else 0.
            cases.append({'output':output,'samples':[{'warmup':j<3,'e2e_ms':value*(1+.02*ri),
                'cycle_ms':(value+restore)*(1+.02*ri)}for j in range(11)]})
        write(run/f'results/measurements/matched_round{ri}_{pid}.json',{'cases':cases,'SYNTHETIC_ONLY':True})
source=real/'code/latency_gate_analysis.py'
spec=importlib.util.spec_from_file_location('latency_fixture',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=night;module.RUN=run;module.OUT=out
write(out/'preregistered.json',{'created_unix':0,'code_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'design_sha256':hashlib.sha256((run/'configs/design.json').read_bytes()).hexdigest(),'guard_multiplier':1.05,
    'fractions':[.95,.8,.65],'metrics':['e2e_ms','cycle_ms'],'SYNTHETIC_ONLY':True})
write(out/'self_checks.json',module.self_check());sys.argv=[str(source)];module.main()
first=(out/'selection.json').read_bytes()
rows=list(csv.DictReader((out/'latency_validation.csv').open()))
assert len(rows)==36 and sum(r['selection_feasible']=='False'for r in rows)==4
for ri in [1,2]:
    for path in(run/'results/measurements').glob(f'matched_round{ri}_*.json'):
        value=json.loads(path.read_text())
        for case in value['cases']:
            for row in case['samples']:
                row['e2e_ms']*=10;row['cycle_ms']*=10
        write(path,value)
module.main();assert(out/'selection.json').read_bytes()==first
rows=list(csv.DictReader((out/'latency_validation.csv').open()))
assert all(int(r['selected_misses'])==16 for r in rows if r['selection_feasible']=='True')
write(root/'VERIFIED_SYNTHETIC_ONLY.json',{'verified':True,'fixture_scope':'Synthetic CPU protocol verification, not experiment results',
    'labels':36,'infeasible_labels':4,'confirmation_perturbation_changes_selection':False,
    'all_feasible_labels_report_all16_failures_after_10x_perturbation':True,
    'analysis_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
print('Synthetic full report passed; later timing changes never alter frozen selection:',root,flush=True)
