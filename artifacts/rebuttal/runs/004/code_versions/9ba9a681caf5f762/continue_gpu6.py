"""Keep the idle card productive after the isolated untraced timing stage."""
import json
import os
import sys
import time

from common import RUN,write_json
from queue_runtime import invoke


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    marker=RUN/'results/runtime_worker.json'
    while True:
        state=json.loads(marker.read_text()) if marker.exists() else {}
        if state.get('status')=='failed':raise RuntimeError('untraced runtime worker failed; inspect before continuation')
        if state.get('status')=='complete':break
        time.sleep(10)
    steps=[('interfaces',[sys.executable,RUN/'code/measure_interfaces.py']),
           ('large_shapes',[sys.executable,RUN/'code/measure_large_shapes.py']),
           ('profiles',[sys.executable,RUN/'code/queue_runtime.py','--profiles']),
           ('mixed_speed',[sys.executable,RUN.parent/'006/code/work_queue.py']),
           ('mixed_quality',[sys.executable,RUN.parent/'006/code/work_queue.py','--quality'])]
    outcomes={}
    for label,command in steps:
        done=RUN/f'results/continuation/{label}.json'
        if done.exists() and json.loads(done.read_text()).get('status')=='complete':continue
        write_json(done,{'status':'running','gpu':6});print(label,'started',flush=True)
        try:
            invoke(command,'continuation_'+label);outcome={'status':'complete','gpu':6}
        except Exception as e:outcome={'status':'failed','error':repr(e),'gpu':6}
        write_json(done,outcome);outcomes[label]=outcome;print(label,outcome,flush=True)
    write_json(RUN/'results/continuation_status.json',outcomes)


if __name__=='__main__':main()
