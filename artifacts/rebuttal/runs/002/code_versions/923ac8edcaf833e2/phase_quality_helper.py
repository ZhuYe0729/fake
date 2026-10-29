"""Use the speed GPU for the two fixed phase controls after timing is finished."""
import json
import fcntl
import os
import time

from common import RUN, write_json
from experiment_queue import worker, release


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    state=RUN/'results/queue_phase_helper.json'
    write_json(state,{'status':'waiting','pid':os.getpid(),'gpu':'3'})
    while json.loads((RUN/'results/queue_primary.json').read_text())['status']!='complete':time.sleep(5)
    write_json(state,{'status':'running','pid':os.getpid(),'gpu':'3'})
    for policy in ['phase_sq_int8_w4a16_ours','phase_fp8_w4a16_ours']:
        worker(policy,'export,decode_nll,generation')
        release(policy)
    points=json.loads((RUN/'configs/evaluation_points.json').read_text())['points']
    controls=sorted([p for p in points if p['family']=='daso5_measured' and p['full']],
                    key=lambda p:(p['scenario']!='prefill_only',-p['quality_budget']))
    for point in controls:
        policy=point['policy_id']
        phase='prefill' if point['scenario']=='prefill_only' else 'decode'
        complete=(RUN/f'results/nll/{policy}_test_{phase}.json').exists()
        if phase=='prefill':
            complete &= all((RUN/f'results/tasks/{policy}/{task}/full/result.json').exists()
                            for task in ['wikitext','winogrande','arc_easy','arc_challenge','mmlu'])
        else:
            complete &= all((RUN/f'results/generation/{policy}/{task}/metrics_0000_end.json').exists()
                            for task in ['cnn_dm_1000','dsum','IWSLT'])
        if complete:continue
        lock_path=RUN/f'results/locks/{policy}.lock'
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
        # The worker reacquires the existing exclusive policy lock. A rare race
        # only waits for an existing worker and then skips completed results.
        worker(policy,'export,nll,tasks' if phase=='prefill' else 'export,decode_nll,generation')
        release(policy)
    write_json(state,{'status':'complete','pid':os.getpid(),'gpu':'3'})


if __name__=='__main__':main()
