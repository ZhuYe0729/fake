"""Freeze the full union of run009 controls and run005 verified choices."""
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    assert not (RUN/'configs/candidate_pool.json').exists()
    old=RUN.parent/'009';verified=RUN.parent/'005'
    for d in ['code','configs/policies','data','reference','results','logs']:(RUN/d).mkdir(parents=True,exist_ok=True)
    for name in ['activation_scales.pt','test_64_2112.pt']:(RUN/'data'/name).symlink_to((old/'data'/name).resolve())
    files=['common.py','eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py','evaluate.py']
    for name in files:shutil.copy2(old/'code'/name,RUN/'code'/name)
    shutil.copytree(old/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    design=read(old/'configs/design.json');design.update({'speed_calibration_seed':144983,
        'scope':'Same-card confirmatory timing of ALL run009 points plus ALL run005 verified quality-budget choices. No new policy/quality selection; inherited identical B8 test protocol.',
        'source_selection_sha256':digest(verified/'configs/selected_policies.json'),'source_hawq_pool_sha256':digest(old/'configs/candidate_pool.json')})
    write(RUN/'configs/design.json',design)
    points=[];policies={};by_signature={};inheritance=[]
    sources=[('009',p) for p in read(old/'configs/candidate_pool.json')['points']]
    sources += [('005',p) for p in read(verified/'configs/selected_policies.json')['points'] if p['family']=='DASO plus validation']
    data_sha=read(old/'results/test/uniform_dense_bf16/prefill_only.json')['data_sha256']
    for rid,point in sources:
        src=RUN.parent/rid;original_pid=point['policy_id'];policy=read(src/f'configs/policies/{original_pid}.json')
        sig=hashlib.sha256(json.dumps(policy['method_map'],sort_keys=True).encode()).hexdigest()
        pid=by_signature.setdefault(sig,original_pid);scenario=point['scenario']
        if pid not in policies:
            write(RUN/f'configs/policies/{pid}.json',{**policy,'policy_id':pid});policies[pid]={'signature':sig,'scenarios':[]}
        quality=src/f'results/test/{original_pid}/{scenario}.json';q=read(quality)
        assert q['data_sha256']==data_sha and q['quality_score_batch']==8 and len(q['blocks'])==64
        target=RUN/f'results/test/{pid}/{scenario}.json'
        if scenario not in policies[pid]['scenarios']:
            policies[pid]['scenarios'].append(scenario);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(quality,target)
            inheritance.append({'target':str(target.relative_to(RUN)),'source':str(quality.resolve()),'sha256':digest(quality)})
        else:assert abs(read(target)['avg_nll']-q['avg_nll'])<1e-6
        points.append({**point,'policy_id':pid,'source_run':rid,'source_policy':original_pid})
    assert len(points)==43 and len(policies)==19
    order=[p for p in policies if p!='uniform_dense_bf16'];random.Random(144985).shuffle(order)
    order=['uniform_dense_bf16']+order+['uniform_dense_bf16_bracket_end']
    bf16=read(RUN/'configs/policies/uniform_dense_bf16.json');write(RUN/'configs/policies/uniform_dense_bf16_bracket_end.json',{**bf16,'policy_id':'uniform_dense_bf16_bracket_end'})
    write(RUN/'configs/candidate_pool.json',{'points':points,'policies':policies,'speed_order':order,
        'frozen_before_new_speed':True,'time_ns':time.time_ns(),'design_sha256':digest(RUN/'configs/design.json')})
    write(RUN/'reference/inherited_quality.json',{'records':inheritance,'data_sha256':data_sha,
        'scope':'Original canonical states/scales and identical runtime methods; exact maps deduplicated. Original quality GPU/time preserved; no new quality labels or retuning.'})
    write(RUN/'reference/inherited_code.json',{name:digest(RUN/'code'/name) for name in files})
    print(json.dumps({'comparison_points':43,'unique_maps':len(policies),'unique_quality_files':len(inheritance),'speed_order':order}))


if __name__=='__main__':main()
