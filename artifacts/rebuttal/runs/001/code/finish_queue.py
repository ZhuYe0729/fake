"""Explicit remaining stages for this single experiment; one process per assigned GPU."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from baseline_worker import TEMP,invoke
from common import RUN,write_json
from checkpoint_evidence import inspect_checkpoint


def command(script,*args):
    return [sys.executable,str(RUN/'code'/script),*map(str,args)]


def worker(policies,actions):
    invoke(command('baseline_worker.py','--policies',','.join(policies),'--actions',actions),
           RUN/f'logs/queue_{os.getpid()}_{time.time_ns()}.log')


def release(policy_id):
    checkpoint=TEMP/'checkpoints'/policy_id
    binary=checkpoint/'model.safetensors'
    if not binary.exists():return
    destination=RUN/'results/export_metadata'/policy_id
    destination.mkdir(parents=True,exist_ok=True)
    inspect_checkpoint(checkpoint,destination)
    for source in checkpoint.glob('*.json'):shutil.copy2(source,destination/source.name)
    size=binary.stat().st_size
    binary.unlink()
    write_json(destination/'temporary_binary_cleanup.json',{'path':str(binary),'bytes':size,
        'reason':'owned reproducible candidate export after recorded evaluations',
        'rebuild':'code/export_model.py with archived policy and canonical states',
        'preserved':'all code, logs, JSON/CSV results and documentation'})


def evaluate_daso(scenario):
    points=json.loads((RUN/f'results/daso/{scenario}/predicted_points.json').read_text())
    if not points:raise ValueError('DASO produced no policy')
    selected=[]
    for budget in [.02,.07,max(p['quality_budget'] for p in points)]:
        selected.append(min(points,key=lambda row:abs(row['quality_budget']-budget))['policy_id'])
    selected=list(dict.fromkeys(selected))
    if len(selected)<3 and len(points)>=3:
        for row in [points[0],points[len(points)//2],points[-1]]:
            if row['policy_id'] not in selected:selected.append(row['policy_id'])
            if len(selected)==3:break
    write_json(RUN/f'configs/selected_daso_{scenario}.json',{'rule':'nearest predicted budgets .02, .07, maximum; no downstream labels',
        'policies':selected})
    for point in points:
        policy_id=point['policy_id']
        worker([policy_id],'export,speed,'+('nll' if scenario=='prefill_only' else 'decode_nll'))
        if policy_id in selected:
            worker([policy_id],'tasks' if scenario=='prefill_only' else 'generation')
        release(policy_id)


def evaluate_hawq():
    allocation=json.loads((RUN/'results/hawq_marlin_int4_allocation.json').read_text())
    chosen={}
    for scenario in ['prefill_only','prefill_decode']:
        points=[p for p in allocation['policies'] if p['scenario']==scenario]
        chosen[scenario]=list(dict.fromkeys([points[0]['policy_id'],points[len(points)//2]['policy_id'],points[-1]['policy_id']]))
    write_json(RUN/'configs/selected_hawq_policies.json',{'rule':'first, middle, last ILP latency-sweep points, before downstream evaluation',
        'policies':chosen})
    for point in allocation['policies']:
        policy_id,scenario=point['policy_id'],point['scenario']
        worker([policy_id],'export,speed,'+('nll' if scenario=='prefill_only' else 'decode_nll'))
        if policy_id in chosen[scenario]:
            worker([policy_id],'tasks' if scenario=='prefill_only' else 'generation')
        release(policy_id)
    # Auxiliary candidate-format control: real speed/NLL only, separately labeled.
    control=json.loads((RUN/'results/hawq_fp4_allocation.json').read_text())
    control_selected={scenario: [p for p in control['policies'] if p['scenario']==scenario][len([p for p in control['policies'] if p['scenario']==scenario])//2]['policy_id']
                      for scenario in ['prefill_only','prefill_decode']}
    write_json(RUN/'configs/selected_hawq_fp4_controls.json',{'rule':'middle latency-sweep point per scenario, before downstream evaluation','policies':control_selected})
    for point in control['policies']:
        worker([point['policy_id']],'export,speed,'+('nll' if point['scenario']=='prefill_only' else 'decode_nll'))
        if point['policy_id']==control_selected[point['scenario']]:
            worker([point['policy_id']],'tasks' if point['scenario']=='prefill_only' else 'generation')
        release(point['policy_id'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--queue',choices=['gpu0','gpu3','gpu4','gpu7'],required=True)
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    state={'queue':args.queue,'pid':os.getpid(),'waiting_for':args.wait_pid,'status':'waiting'}
    target=RUN/f'results/queue_{args.queue}.json'
    write_json(target,state)
    if args.wait_pid:
        while Path(f'/proc/{args.wait_pid}').exists():time.sleep(10)
    state['status']='running';write_json(target,state)
    try:
        if args.queue=='gpu0':
            worker(['uniform_dense_bf16','uniform_dense_nvfp4'],'speed,nll,decode_nll,tasks')
            invoke(command('daso_pipeline.py','calibrate','--scenario','prefill_decode'),RUN/'logs/daso_calibration_decode_01.log')
            invoke(command('daso_pipeline.py','fit','--scenario','prefill_decode'),RUN/'logs/daso_fit_decode_01.log')
            evaluate_daso('prefill_decode')
            worker(['uniform_dense_bf16','uniform_dense_nvfp4'],'generation')
        elif args.queue=='gpu3':
            invoke(command('daso_pipeline.py','fit','--scenario','prefill_only'),RUN/'logs/daso_fit_prefill_01.log')
            evaluate_daso('prefill_only')
            invoke(command('common_card_speed.py'),RUN/'logs/common_card_speed_01.log')
        elif args.queue=='gpu4':
            smoke=RUN/'results/generation/diagnostic_generation_smoke_bf16/cnn_dm_1000/metrics_0000_16.json'
            if not smoke.exists():
                invoke(command('generate_tasks.py','--policy','diagnostic_generation_smoke_bf16',
                    '--checkpoint',TEMP/'checkpoints/uniform_dense_bf16','--dataset','cnn_dm_1000','--end','16'),
                    RUN/'logs/generation_smoke_bf16.log')
            evaluate_hawq()
            worker(['uniform_marlin_int4'],'generation')
            worker(['phase_uniform_nvfp4_marlin'],'export,speed,decode_nll,generation')
        else:
            worker(['uniform_sparse_bf16','uniform_sparse_nvfp4','uniform_w4a16_ours'],'speed,nll,decode_nll,tasks,generation')
            worker(['uniform_int8'],'speed,nll,decode_nll,tasks,generation')
        state['status']='complete'
    except Exception as exc:
        state.update(status='failed',error=repr(exc));write_json(target,state);raise
    write_json(target,state)


if __name__=='__main__':main()
