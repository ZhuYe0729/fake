"""Apply only the preregistered validation/calibration-speed selection rules."""
import hashlib
import json

import numpy as np

from common import RUN,write_json


def main():
    assert not (RUN/'configs/selected_policies.json').exists(),'selection is immutable'
    pool=json.loads((RUN/'configs/candidate_pool.json').read_text());design=json.loads((RUN/'configs/design.json').read_text())
    assert not list((RUN/'results/test').glob('*/*.json')) and not list((RUN/'results/test_speed').glob('*/*.json'))
    assert all((RUN/f'results/validation_complete/{p}.json').exists() and (RUN/f'results/calibration_speed_complete/{p}.json').exists() for p in pool['policies'])
    evaluated={};validation=[]
    for scenario in ['prefill_only','prefill_decode']:
        reference=json.loads((RUN/f'results/validation/uniform_dense_bf16/{scenario}.json').read_text())
        baseline=json.loads((RUN/f'results/calibration_speed/uniform_dense_bf16/{scenario}.json').read_text())['median_ms']
        for pid,job in pool['policies'].items():
            if scenario not in job['scenarios']:continue
            r=json.loads((RUN/f'results/validation/{pid}/{scenario}.json').read_text())
            speed=json.loads((RUN/f'results/calibration_speed/{pid}/{scenario}.json').read_text())['median_ms']
            differences=np.array([a['avg_nll']-b['avg_nll'] for a,b in zip(r['blocks'],reference['blocks'],strict=True)])
            mean=float(differences.mean());se=float(differences.std(ddof=1)/np.sqrt(len(differences)))
            guard=0. if pid=='uniform_dense_bf16' else mean+design['quality_guard']['z']*se+design['quality_guard']['additive_delta_nll']
            value={'policy_id':pid,'scenario':scenario,'delta_nll':mean,'paired_se':se,'guarded_delta_nll':guard,
                'calibration_ms':speed,'guarded_calibration_ms':speed*(1+design['latency_guard_fraction']),
                'bf16_calibration_ms':baseline}
            evaluated[scenario,pid]=value;validation.append(value)
    selected=[]
    # These original/solver points were specified before validation; selection
    # does not remove unfavorable points from the final comparison.
    for point in pool['points']:
        keep=(point['family']=='uniform' or point['family']=='exact inverse' or
            point['family'] in ['exact forward','original DP'] and point.get('quality_budget') in design['quality_budgets'])
        if keep:selected.append({**point,'selection':'preregistered raw/anchor comparator'})
    for scenario in ['prefill_only','prefill_decode']:
        values=[v for (s,p),v in evaluated.items() if s==scenario]
        for budget in design['quality_budgets']:
            feasible=[v for v in values if v['guarded_delta_nll']<=budget]
            winner=min(feasible,key=lambda v:(v['calibration_ms'],v['policy_id']))
            selected.append({'label':f'verified_{scenario}_q{budget:g}','policy_id':winner['policy_id'],
                'scenario':scenario,'family':'DASO plus validation','quality_budget':budget,
                'selection':'preregistered guarded validation feasibility then minimum calibration latency',
                'validation':winner,'feasible_candidates':len(feasible)})
        for fraction in design['inverse_latency_fractions']:
            baseline=values[0]['bf16_calibration_ms'];cap=baseline*fraction
            feasible=[v for v in values if v['guarded_calibration_ms']<=cap]
            if feasible:
                winner=min(feasible,key=lambda v:(v['guarded_delta_nll'],v['calibration_ms'],v['policy_id']))
                selected.append({'label':f'verified_inverse_{scenario}_r{fraction:g}','policy_id':winner['policy_id'],
                    'scenario':scenario,'family':'inverse plus validation','actual_latency_sla_ms':cap,
                    'selection':'preregistered 5% guarded calibration latency then minimum guarded validation quality',
                    'validation':winner,'feasible_candidates':len(feasible)})
            else:
                selected.append({'label':f'verified_inverse_{scenario}_r{fraction:g}','policy_id':None,
                    'scenario':scenario,'family':'inverse plus validation','actual_latency_sla_ms':cap,
                    'infeasible':True,'feasible_candidates':0})
    policies={}
    for point in selected:
        pid=point['policy_id']
        if pid is None:continue
        policies.setdefault(pid,{'scenarios':[],'signature':pool['policies'][pid]['signature']})
        if point['scenario'] not in policies[pid]['scenarios']:policies[pid]['scenarios'].append(point['scenario'])
    write_json(RUN/'results/validation_selection_audit.json',{'candidates':validation,'selected_points':selected,
        'rules_sha256':hashlib.sha256((RUN/'configs/design.json').read_bytes()).hexdigest(),
        'test_results_present_before_selection':list((RUN/'results/test').glob('*/*.json')) if (RUN/'results/test').exists() else []})
    write_json(RUN/'configs/selected_policies.json',{'points':selected,'policies':policies,
        'pool_sha256':hashlib.sha256((RUN/'configs/candidate_pool.json').read_bytes()).hexdigest(),
        'rules_sha256':hashlib.sha256((RUN/'configs/design.json').read_bytes()).hexdigest(),
        'selection_audit_sha256':hashlib.sha256((RUN/'results/validation_selection_audit.json').read_bytes()).hexdigest(),
        'frozen_before_test':True})
    print(json.dumps({'selected_points':len(selected),'unique_policies':len(policies),
        'infeasible_inverse_requests':sum(p.get('infeasible',False) for p in selected)}))


if __name__=='__main__':main()
