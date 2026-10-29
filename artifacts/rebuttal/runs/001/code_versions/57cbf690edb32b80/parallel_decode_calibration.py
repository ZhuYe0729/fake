"""Finish disjoint decode labels on newly free GPUs, then resume the original queue."""
import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import RUN,write_json


def done(pid):
    path=Path(f'/proc/{pid}/stat')
    return not path.exists() or path.read_text().rsplit(') ',1)[1].startswith('Z')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-pid',required=True,type=int)
    args=parser.parse_args();parent=args.parent_pid
    command=Path(f'/proc/{parent}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'daso_pipeline.py' in command and 'calibrate' in command and 'prefill_decode' in command
    children_path=Path(f'/proc/{parent}/task/{parent}/children')
    paused=False;workers=[]
    state={'original_parent':parent,'status':'preparing','workers':[]}
    target=RUN/'results/parallel_decode_calibration.json'
    try:
        while not paused:
            for value in children_path.read_text().split():
                pid=int(value)
                path=Path(f'/proc/{pid}/cmdline')
                if not path.exists():continue
                child=path.read_bytes().replace(b'\0',b' ').decode()
                match=re.search(r'cal_prefill_decode_p(\d\d)',child)
                if not match:continue
                os.kill(parent,signal.SIGSTOP)
                time.sleep(.05)
                if str(pid) in children_path.read_text().split():
                    paused=True;current=int(match.group(1));break
                os.kill(parent,signal.SIGCONT)
            if not paused:time.sleep(.05)
        state.update(status='parent_parked',current_policy=current,current_child=pid,
                     reason='resume only after disjoint later ranges finish; no duplicate labels or exports')
        write_json(target,state)
        while not done(pid):time.sleep(1)
        begin=current+1
        for index,gpu in enumerate([0,1,2]):
            first=begin+(72-begin)*index//3;end=begin+(72-begin)*(index+1)//3
            if first==end:continue
            command=[sys.executable,str(RUN/'code/daso_pipeline.py'),'calibrate','--scenario','prefill_decode',
                     '--begin',str(first),'--end',str(end)]
            log=RUN/f'logs/parallel_decode_gpu{gpu}_{first:02d}_{end:02d}_{time.time_ns()}.log'
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            with log.open('x') as handle:
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=environment)
            workers.append(process)
            state['workers'].append({'gpu':gpu,'begin':first,'end':end,'pid':process.pid,'log':str(log),'command':command})
        state['status']='running';write_json(target,state)
        while any(process.poll() is None for process in workers):time.sleep(2)
        state['returncodes']=[process.returncode for process in workers]
        state['status']='complete' if all(process.returncode==0 for process in workers) else 'sequential_fallback'
        # An unsuccessful export must be preserved under another name before
        # the original process retries missing policies without any concurrency.
        for index in range(begin,72):
            result=RUN/f'results/calibration/prefill_decode/cal_prefill_decode_p{index:02d}.json'
            if result.exists():continue
            checkpoint=Path('/dev/shm/daso_rebuttal_001_20261025/calibration_checkpoints')/f'cal_prefill_decode_p{index:02d}'
            if checkpoint.exists():checkpoint.rename(checkpoint.with_name(checkpoint.name+f'_parallel_attempt_{time.time_ns()}'))
    finally:
        for process in workers:process.wait()
        if paused:
            os.kill(parent,signal.SIGCONT)
            state['original_parent_resumed']=True
        write_json(target,state)
    print(state,flush=True)


if __name__=='__main__':main()
