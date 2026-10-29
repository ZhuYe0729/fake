"""Continue validation/selection/final testing without an overnight idle gap."""
import json
import os
import subprocess
import sys
import time

from common import RUN,write_json


def stage(name,jobs):
    processes=[];handles=[]
    for label,gpu,command in jobs:
        log=RUN/f'logs/pipeline_{label}_{time.time_ns()}.log';handle=log.open('w');handles.append(handle)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)}
        process=subprocess.Popen(list(map(str,command)),stdout=handle,stderr=subprocess.STDOUT,env=env)
        processes.append((label,process,log));print(name,label,'started',process.pid,'gpu',gpu,flush=True)
    try:
        while any(p.poll() is None for _,p,_ in processes):
            write_json(RUN/'results/pipeline.json',{'status':'running','stage':name,
                'workers':{label:{'pid':p.pid,'returncode':p.poll(),'log':str(log)} for label,p,log in processes}})
            failed=[(label,p.poll(),str(log)) for label,p,log in processes if p.poll() not in [None,0]]
            if failed:raise RuntimeError(str(failed))
            time.sleep(5)
        assert all(p.returncode==0 for _,p,_ in processes),[(label,p.returncode) for label,p,_ in processes]
    finally:
        for handle in handles:handle.close()


def main():
    marker=RUN.parent/'004/results/runtime_worker.json'
    while True:
        state=json.loads(marker.read_text()) if marker.exists() else {}
        if state.get('status')=='failed':raise RuntimeError('inspect failed run004 timing first')
        if state.get('status')=='complete':break
        time.sleep(10)
    try:
        if not (RUN/'configs/selected_policies.json').exists():
            stage('calibration_and_validation',[
                ('calibration_speed',2,[sys.executable,RUN/'code/work_queue.py','--role','speed']),
                ('validation_gpu4',4,[sys.executable,RUN/'code/work_queue.py','--role','validation']),
                ('validation_gpu7',7,[sys.executable,RUN/'code/work_queue.py','--role','validation'])])
            stage('selection',[('select_verified','2',[sys.executable,RUN/'code/select_verified.py'])])
        stage('final_test',[
            ('test_speed',2,[sys.executable,RUN/'code/work_queue.py','--role','test_speed']),
            ('test_gpu4',4,[sys.executable,RUN/'code/work_queue.py','--role','test']),
            ('test_gpu7',7,[sys.executable,RUN/'code/work_queue.py','--role','test'])])
        write_json(RUN/'results/pipeline.json',{'status':'complete'})
        print('All preregistered selection and final tests complete',flush=True)
    except Exception as e:write_json(RUN/'results/pipeline.json',{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
