"""Give the main validation/serving/pruning experiments resource priority."""
import json
import os
import subprocess
import sys
import time

from common import RUN,write_json


def main():
    while not (RUN.parent/'005/configs/selected_policies.json').exists():time.sleep(10)
    processes=[];handles=[]
    for role,gpu in [('speed',0),('quality',1)]:
        f=(RUN/f'logs/pipeline_{role}_{time.time_ns()}.log').open('w');handles.append(f)
        process=subprocess.Popen([sys.executable,str(RUN/'code/work_queue.py'),'--role',role],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=f,stderr=subprocess.STDOUT)
        processes.append((role,process));print(role,process.pid,gpu,'started',flush=True)
    try:
        while any(p.poll() is None for _,p in processes):
            failed=[(role,p.poll()) for role,p in processes if p.poll() not in [None,0]]
            if failed:raise RuntimeError(str(failed))
            write_json(RUN/'results/pipeline.json',{'status':'running','workers':{r:{'pid':p.pid,'returncode':p.poll()} for r,p in processes}})
            time.sleep(5)
        assert all(p.returncode==0 for _,p in processes)
        write_json(RUN/'results/pipeline.json',{'status':'complete'})
    except Exception as e:write_json(RUN/'results/pipeline.json',{'status':'failed','error':repr(e)});raise
    finally:
        for f in handles:f.close()


if __name__=='__main__':main()
