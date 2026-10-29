"""Check inverse Pareto consistency and quantify the original DP grid loss."""
import json
import math

import numpy as np

from common import RUN,write_json
from exact_solver import solve


def main():
    pool=json.loads((RUN/'configs/candidate_pool.json').read_text());checks=[];grid=[]
    for scenario in ['prefill_only','prefill_decode']:
        data=json.loads((RUN/f'data/actions_{scenario}.json').read_text())
        q=np.array(data['quality']);t=np.array(data['latency_ms']);names=data['names'];phases=data['phases'];bias=data['bias']
        maximum=float(q.max(1).sum());quantum=maximum/2000
        sq=data['methods'].index('sq_int8')
        pair_q=q[:len(names),sq]+(q[len(names):,sq] if len(phases)==2 else 0.)
        rounded=sum(0 if v<=0 else max(1,math.ceil(v/quantum)) for v in pair_q)*quantum
        grid.append({'scenario':scenario,'quantum':quantum,'uniform_sq_int8_raw_predicted_delta_nll':bias+float(q[:,sq].sum()),
            'uniform_sq_int8_rounded_predicted_delta_nll':bias+rounded,
            'rounding_upper_bound':len(names)*quantum,
            'scope':'Exactly the original DP: one rounded action per module; P+D rounds the phase-pair sum once.'})
        for point in pool['points']:
            if point['scenario']!=scenario or point['family']!='exact forward':continue
            result=solve(q,t,point['predicted_linear_ms']+1e-7,inverse=True)
            assert bias+result['quality']<=point['predicted_delta_nll']+1e-8
            forward=solve(q,t,result['quality']+1e-10)
            assert forward['latency_ms']<=point['predicted_linear_ms']+1e-6
            checks.append({'label':point['label'],'passed':True,'inverse_quality':bias+result['quality'],
                'forward_quality':point['predicted_delta_nll'],'latency_ms':point['predicted_linear_ms'],
                'inverse_solver_seconds':result['seconds'],'forward_solver_seconds':forward['seconds']})
    write_json(RUN/'results/surrogate_audit.json',{'passed':True,'checks':checks,'grid_examples':grid,
        'scope':'Identical separable surrogate, exact real-valued quality costs in MILP. Forward/inverse Pareto equivalence does not guarantee actual NLL or actual latency.'})
    print(json.dumps({'passed':True,'pareto_checks':len(checks),'grid':grid}))


if __name__=='__main__':main()
