"""Audit original module-pair DP rounding against identical exact matrices."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
REPO=RUN.parents[3]
sys.path[:0]=[str(REPO/'daso/src'),str(RUN.parent/'017/code')]
import numpy as np
from daso.optimization.solver import Action,ModuleChoices,solve_budget
from exact_solver import solve


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    design=read(RUN/'configs/design.json')
    assert sha(REPO/'daso/src/daso/optimization/solver.py')==design['dp_source_sha256']
    rows,inverse=[],[]
    for rid in design['models']:
        source=RUN.parent/rid
        for name,digest in design['sources_sha256'][rid].items():assert sha(source/name)==digest
        frozen=read(source/'configs/frozen_design.json')
        profile={(r['module_type'],r['method'],int(r['m'])):float(r['latency_ms'])
            for r in csv.DictReader((source/'reference/graph_action_profile.csv').open())}
        for scene in design['scenarios']:
            data=read(source/f'data/actions_{scene}.json')
            q=np.array(data['quality']);methods=data['methods'];names=data['names'];phases=data['phases']
            t=np.array([[profile[n.rsplit('.',1)[-1],method,16384 if phase=='prefill' else 8]*
                (1 if phase=='prefill' else 63) for method in methods] for phase in phases for n in names])
            modules=[]
            for i,name in enumerate(names):
                actions=[]
                for a,pre in enumerate(methods):
                    for b in (range(len(methods)) if len(phases)==2 else [a]):
                        quality=q[i,a]+(q[i+len(names),b] if len(phases)==2 else 0.)
                        latency=t[i,a]+(t[i+len(names),b] if len(phases)==2 else 0.)
                        actions.append(Action(pre,methods[b],float(quality),float(latency)))
                modules.append(ModuleChoices(name,tuple(actions)))
            for budget in design['budgets']:
                cap=budget-data['bias']
                exact=solve(q,t,cap)
                assert exact['optimal'] and exact['quality']<=cap+1e-9
                back=solve(q,t,exact['latency_ms']+1e-6,inverse=True)
                assert back['quality']<=exact['quality']+1e-8
                inverse.append({'run':rid,'scenario':scene,'budget':budget,'passed':True,
                    'forward_quality':exact['quality']+data['bias'],
                    'inverse_quality':back['quality']+data['bias'],'latency_cap_ms':exact['latency_ms'],
                    'inverse_latency_ms':back['latency_ms'],'inverse_seconds':back['seconds']})
                for bins in design['dp_bins']:
                    started=time.perf_counter()
                    dp=solve_budget(modules,cap,bins=bins)
                    seconds=time.perf_counter()-started
                    assert dp.quality_cost<=cap+1e-9
                    assert dp.latency_ms+1e-6>=exact['latency_ms']
                    rows.append({'run':rid,'scenario':scene,'budget':budget,'bins':bins,
                        'dp_quality':dp.quality_cost+data['bias'],'exact_quality':exact['quality']+data['bias'],
                        'dp_linear_ms':dp.latency_ms,'exact_linear_ms':exact['latency_ms'],
                        'dp_latency_gap_percent':(dp.latency_ms/exact['latency_ms']-1)*100,
                        'dp_seconds':seconds,'milp_seconds':exact['seconds'],
                        'rounding_quantum':sum(max(a.quality_cost for a in m.actions) for m in modules)/bins})
                    write(RUN/'results/progress.json',{'run':rid,'scenario':scene,'budget':budget,'bins':bins,'completed':len(rows)})
                    print(rows[-1],flush=True)
    with (RUN/'results/solver_points.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(RUN/'results/inverse_checks.json',inverse)
    write(RUN/'results/final_audit.json',{'complete':True,'verified':True,'points':len(rows),'inverse_checks':len(inverse),
        'design_sha256':sha(RUN/'configs/design.json'),'code_sha256':sha(Path(__file__)),
        'csv_sha256':sha(RUN/'results/solver_points.csv'),'inverse_sha256':sha(RUN/'results/inverse_checks.json'),
        'scope':'Fixed additive surrogate only. No claim of real-model accuracy/latency guarantees.'})
    write(RUN/'results/worker.json',{'status':'complete','stage':'solver_audit','points':len(rows)})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/'results/worker.json',{'status':'failed','error':repr(error)});raise
