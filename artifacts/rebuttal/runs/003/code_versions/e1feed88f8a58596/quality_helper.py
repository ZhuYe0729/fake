"""Use otherwise idle identical GPUs for the shared quality queue."""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from common import RUN, write_json
from orchestrate import invoke, release


def main():
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu in ['2','6']
    design=json.loads((RUN/'configs/evaluation_points.json').read_text())
    name='helper'+gpu;state=RUN/f'results/worker_{name}.json'
    write_json(state,{'status':'running','pid':os.getpid(),'gpu':gpu,'scope':'quality only; identical RTX5090; shared locks'})
    try:
        while True:
            pending=0;claimed=False
            for policy,job in design['policies'].items():
                done=RUN/f'results/quality_complete/{policy}.json'
                if done.exists():continue
                pending+=1;ready=RUN/f'results/speed_complete/{policy}.json'
                if not ready.exists():continue
                lock=RUN/f'results/locks/{policy}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
                with lock.open('a') as handle:
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if done.exists():continue
                    model=Path(json.loads(ready.read_text())['checkpoint'])
                    assert (model/'model.safetensors').exists()
                    print('quality started',policy,len(job['workloads']),flush=True)
                    invoke([sys.executable,RUN/'code/measure_matrix.py','--checkpoint',model,
                        '--policy',policy,'--mode','quality','--workloads',','.join(job['workloads'])],policy+'_quality_'+name)
                    write_json(done,{'complete':True,'policy':policy,'worker':name});release(policy)
                    print('quality complete',policy,flush=True);claimed=True;break
            if not pending:break
            if not claimed:time.sleep(5)
        write_json(state,{'status':'complete','pid':os.getpid(),'gpu':gpu})
    except Exception as error:
        write_json(state,{'status':'failed','pid':os.getpid(),'error':repr(error)});raise


if __name__=='__main__':main()
