"""Freeze the small reproducibility study before measuring new results."""
import hashlib
import json
import shutil
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert not (RUN/'configs/design.json').exists()
    old=RUN.parent/'011'
    for d in ['configs/policies','data','reference','results','logs']:(RUN/d).mkdir(parents=True,exist_ok=True)
    for name in ['common.py','eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py']:shutil.copy2(old/'code'/name,RUN/'code'/name)
    shutil.copytree(old/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    (RUN/'data/activation_scales.pt').symlink_to((old/'data/activation_scales.pt').resolve())
    policies=['uniform_dense_bf16','transfer_daso5_measured_prefill_decode_04',
        'hawq_fp4_phase_pair_signed_prefill_decode_q0.01','hawq_fp4_phase_pair_signed_prefill_decode_q0.07',
        'inverse_prefill_decode_r0.65','exact_prefill_decode_q0.02']
    sources={};qualities=[]
    for pid in policies:
        origin='005' if pid=='exact_prefill_decode_q0.02' else '011'
        p=RUN.parent/origin/f'configs/policies/{pid}.json';shutil.copy2(p,RUN/f'configs/policies/{pid}.json')
        sources[pid]={'run':origin,'policy_sha256':digest(p)}
        q=RUN.parent/origin/f'results/test/{pid}/prefill_decode.json'
        qualities.append({'policy':pid,'source':str(q.resolve()),'sha256':digest(q),'delta_reused_only_for_reporting':True})
    original=read(RUN.parent/'005/results/validation_selection_audit.json')
    candidates=['uniform_dense_bf16','inverse_prefill_decode_r0.65','exact_prefill_decode_q0.02']
    guards=[v for v in original['candidates'] if v['scenario']=='prefill_decode' and v['policy_id'] in candidates]
    assert len(guards)==3
    orders=[policies,list(reversed(policies)),policies[2:]+policies[:2],list(reversed(policies[2:]+policies[:2]))]
    write(RUN/'configs/design.json',{'policies':policies,'sources':sources,'gpu':0,'warmups':3,'iterations':8,
        'batch':8,'input':2048,'outputs':[1,64],'kv_bytes':6*1024**3,
        'rounds':[{'round':i,'split':'calibration' if i<2 else 'test','seed':144983 if i<2 else 144995,'order':order} for i,order in enumerate(orders)],
        'numa':{'local':{'node':0,'cpus':'0-7'},'remote':{'node':1,'cpus':'32-39'}},
        'remote_policies':[policies[1],policies[3]],'remote_rounds':[0,1],
        'copy_probe':{'order':[0,1,1,0],'mib':[64,512],'warmups':5,'iterations':20},
        'sla':{'candidates':candidates,'original_validation':guards,'fractions':[.95,.8,.65],
            'rules':{'original_5_percent':'median(all calibration timed)*1.05','empirical_upper':'max(all calibration timed)*1.02'},
            'selection':'minimum ORIGINAL validation guarded_delta_nll among latency-feasible candidates; no final quality/test latency used',
            'freeze_before_round':2,'scope':'empirical reliability under stated conditions, not a hard real-time bound'},
        'registered_time_ns':time.time_ns(),'scope':'Diagnose timing stability/NUMA; fixed maps and unchanged backends, no new quality labels or quality fitting.'})
    write(RUN/'reference/inherited_quality.json',{'records':qualities,'original_validation_source':str((RUN.parent/'005/results/validation_selection_audit.json').resolve()),
        'original_validation_sha256':digest(RUN.parent/'005/results/validation_selection_audit.json')})
    write(RUN/'reference/protected_manifests.json',{rid:digest(RUN.parent/rid/'results/manifest.json') for rid in [f'{i:03}' for i in range(1,12)]})
    print(json.dumps({'policies':len(policies),'local_model_processes':24,'remote_model_processes':4,'fresh_quality_labels':0}))


if __name__=='__main__':main()
