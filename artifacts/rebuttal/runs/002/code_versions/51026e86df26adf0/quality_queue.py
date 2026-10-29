"""Claim independent policies dynamically so long full tasks stay balanced."""
import argparse
import fcntl
import json
import os
import time

from common import RUN, write_json
from experiment_queue import worker, release


def complete(point):
    policy=point['policy_id'];phase='prefill' if point['scenario']=='prefill_only' else 'decode'
    if not (RUN/f'results/nll/{policy}_test_{phase}.json').exists():return False
    if not point['full']:return True
    if phase=='prefill':
        return all((RUN/f'results/tasks/{policy}/{task}/full/result.json').exists()
                   for task in ['wikitext','winogrande','arc_easy','arc_challenge','mmlu'])
    return all((RUN/f'results/generation/{policy}/{task}/metrics_0000_end.json').exists()
               for task in ['cnn_dm_1000','dsum','IWSLT'])


def nll_complete(point):
    phase='prefill' if point['scenario']=='prefill_only' else 'decode'
    return (RUN/f'results/nll/{point["policy_id"]}_test_{phase}.json').exists()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--wait-primary',action='store_true');args=parser.parse_args()
    state=RUN/f'results/queue_{args.name}.json'
    def status(value,**extra):write_json(state,{'status':value,'pid':os.getpid(),
        'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'scheduler':'shared policy claims',**extra})
    status('waiting')
    if args.wait_primary:
        while json.loads((RUN/'results/queue_primary.json').read_text())['status']!='complete':time.sleep(5)
    while not (RUN/'configs/evaluation_points.json').exists():time.sleep(5)
    points=json.loads((RUN/'configs/evaluation_points.json').read_text())['points']
    points=sorted(points,key=lambda p:(p['scenario']!='prefill_only',p['family']!='daso8',p['policy_id']))
    jobs=[(p,'nll') for p in points]+[(p,'full') for p in points if p['full']]
    claims=RUN/'results/claims';claims.mkdir(exist_ok=True)
    status('running')
    while not all(complete(p) for p in points):
        worked=False
        for point,stage in jobs:
            policy=point['policy_id']
            done=nll_complete(point) if stage=='nll' else complete(point)
            if done or not (RUN/f'results/speed_complete/{policy}.json').exists():continue
            if stage=='full' and not nll_complete(point):continue
            with (claims/f'{policy}.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (nll_complete(point) if stage=='nll' else complete(point)):continue
                phase='prefill' if point['scenario']=='prefill_only' else 'decode'
                action=('nll' if phase=='prefill' else 'decode_nll') if stage=='nll' else ('tasks' if phase=='prefill' else 'generation')
                actions='export,'+action
                status('running',policy=policy,stage=stage)
                try:
                    worker(policy,actions);release(policy)
                except Exception as exc:
                    status('failed',policy=policy,error=repr(exc));raise
                worked=True
        if not worked:time.sleep(5)
    status('complete')


if __name__=='__main__':main()
