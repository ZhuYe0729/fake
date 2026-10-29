"""Finite experiment queues; one queue per assigned physical GPU."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

from baseline_worker import TEMP, invoke
from checkpoint_evidence import inspect_checkpoint
from common import MODEL, RUN, VLLM, WRAPPER, write_json


def call(script,*args,label):
    invoke([sys.executable,str(RUN/'code'/script),*map(str,args)],RUN/f'logs/{label}.log')


def wait(path):
    while not path.exists():time.sleep(5)


def worker(policy,actions):
    call('baseline_worker.py','--policies',policy,'--actions',actions,label=policy+'_'+actions.replace(',','_'))


def release(policy):
    checkpoint=TEMP/'checkpoints'/policy
    if checkpoint.is_symlink():return
    binary=checkpoint/'model.safetensors'
    if not binary.exists():return
    evidence=RUN/f'results/export_metadata/{policy}'
    evidence.mkdir(parents=True,exist_ok=True)
    inspect_checkpoint(checkpoint,evidence)
    for source in checkpoint.glob('*.json'):shutil.copy2(source,evidence/source.name)
    size=binary.stat().st_size;binary.unlink()
    write_json(evidence/'temporary_binary_cleanup.json',{'path':str(binary),'bytes':size,
        'reason':'owned reproducible export after all assigned evaluations',
        'rebuild':'code/export_model.py with archived policy, activation scales and canonical states',
        'preserved':'all code, logs, JSON/CSV results and documents'})


def speed(policy):
    output=RUN/f'results/speed_common_gpu3/{policy}'
    if (output/'summary.csv').exists():return
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    call('run_workflow.py','speed','--checkpoint',TEMP/'checkpoints'/policy,
        '--scenario','prefill_decode','--output-dir',output,'--vllm-root',VLLM,
        '--cutlass-wrapper-path',WRAPPER,'--warmup-iters','2','--iters','5','--reuse-llm',
        '--gpu-memory-utilization','.85',label=policy+'_common_speed')


def design_evaluation():
    points=[]
    for family in ['daso8','daso5_measured']:
        for scenario in ['prefill_only','prefill_decode']:
            result=json.loads((RUN/f'results/{family}/{scenario}/predicted_points.json').read_text())
            budgets=[.02,.07,.45] if scenario=='prefill_only' else [.02,.07,.22]
            selected={min(result,key=lambda p:abs(p['quality_budget']-budget))['policy_id'] for budget in budgets}
            points.extend([{**p,'family':family,'scenario':scenario,'full':p['policy_id'] in selected} for p in result])
    names=list(json.loads((RUN/'configs/policies/uniform_dense_bf16.json').read_text())['method_map'])
    # Fixed simple phase controls; choices are specified before test evaluation.
    for pre,dec in [('dense_nvfp4','w4a16_ours'),('fp8','w4a16_ours'),
                    ('sq_int8','w4a16_ours'),('sq_int8','dense_bf16'),
                    ('fp8','fp8'),('fp8','sq_int8'),('dense_nvfp4','fp8')]:
        if pre==dec:continue
        policy=f'phase_{pre}_{dec}'
        write_json(RUN/f'configs/policies/{policy}.json',{'policy_id':policy,'scenario':'prefill_decode',
            'default_prefill_method':pre,'default_decode_method':dec,'modules_to_not_convert':['lm_head'],
            'method_map':{name:{'prefill_method':pre,'decode_method':dec} for name in names}})
        points.append({'policy_id':policy,'family':'phase-uniform','scenario':'prefill_decode',
                       'full':(pre,dec) in [('fp8','w4a16_ours'),('sq_int8','w4a16_ours')]})
    write_json(RUN/'configs/evaluation_points.json',{'rule':'fixed budget representatives and fixed phase controls; no test selection','points':points})
    return points


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['cal0','cal1','cal2','primary'],required=True)
    args=parser.parse_args();state=RUN/f'results/queue_{args.role}.json'
    write_json(state,{'status':'waiting','pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    wait(RUN/'results/kernel_audit.json')
    if args.role in ['cal0','primary']:wait(RUN/'results/action_profile_final.json')
    write_json(state,{'status':'running','pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    if args.role.startswith('cal'):
        slot=int(args.role[-1]);begin=72+slot*16
        for scenario in ['prefill_only','prefill_decode']:
            call('daso_pipeline.py','calibrate','--scenario',scenario,'--begin',begin,'--end',begin+16,
                 label=f'calibration_{scenario}_slot{slot}')
        write_json(RUN/f'results/calibration_slot{slot}_complete.json',{'complete':True})
        wait(RUN/'configs/evaluation_points.json')
        points=json.loads((RUN/'configs/evaluation_points.json').read_text())['points']
        # Same-GPU speed is finished before quality workers can export/release a policy.
        for index,p in enumerate(points):
            if index%3!=slot:continue
            policy=p['policy_id'];wait(RUN/f'results/speed_complete/{policy}.json')
            actions='export,'+('nll' if p['scenario']=='prefill_only' else 'decode_nll')
            if p['full']:actions+=',tasks' if p['scenario']=='prefill_only' else ',generation'
            worker(policy,actions);release(policy)
        baseline=['uniform_sq_int8','uniform_fp8','uniform_sq_fp8'][slot]
        wait(RUN/f'results/speed_complete/{baseline}.json')
        worker(baseline,'export,nll,decode_nll,tasks,generation');release(baseline)
    else:
        for policy in ['uniform_dense_bf16','uniform_dense_nvfp4','uniform_w4a16_ours',
                       'uniform_sq_int8','uniform_fp8','uniform_sq_fp8']:
            worker(policy,'export');speed(policy)
            write_json(RUN/f'results/speed_complete/{policy}.json',{'complete':True})
        worker('uniform_dense_bf16','nll,decode_nll')
        for slot in range(3):wait(RUN/f'results/calibration_slot{slot}_complete.json')
        for family in ['extended','restricted']:
            for scenario in ['prefill_only','prefill_decode']:
                extra=['--restricted'] if family=='restricted' else []
                call('daso_pipeline.py','fit','--scenario',scenario,*extra,label=f'fit_{family}_{scenario}')
        points=design_evaluation()
        write_json(RUN/'configs/common_card_speed.json',{'physical_gpu':3,
            'policies':[p['policy_id'] for p in points],
            'rule':'sequential 2 warmups + 5 timed iterations with fixed 3GiB BF16 KV'})
        for p in points:
            policy=p['policy_id'];worker(policy,'export');speed(policy)
            # Releasing before signalling avoids races with quality worker re-export.
            release(policy)
            write_json(RUN/f'results/speed_complete/{policy}.json',{'complete':True})
    write_json(state,{'status':'complete','pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})


if __name__=='__main__':main()
