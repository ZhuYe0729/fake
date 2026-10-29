"""Observe a sustained free window without allocating or changing GPU state."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'runs/034/code'))
from telemetry import nvml


def main():
    since={};previous=None
    while time.time()<1793148789:
        now=time.time();usage={gpu:nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used
            for gpu in range(nvml.nvmlDeviceGetCount())}
        free=[g for g,v in usage.items()if v<512*1024**2]
        for gpu in usage:
            if gpu in free:since.setdefault(gpu,now)
            else:since.pop(gpu,None)
        status={'observed_unix':now,'memory_mib':{g:round(v/1024**2)for g,v in usage.items()},
            'continuous_free_seconds':{g:round(now-s)for g,s in since.items()},'required_free_seconds':120}
        temp=ROOT/'resource_watch.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(ROOT/'resource_watch.json')
        if free!=previous:
            previous=free;print(time.strftime('%H:%M:%S',time.gmtime()),'Free cards',free,'MiB',status['memory_mib'],flush=True)
            with(ROOT/'resource_history.jsonl').open('a')as handle:handle.write(json.dumps(status)+'\n')
        ready=[g for g in [*range(1,len(usage)),0]if g in since and now-since[g]>=120]
        if ready:
            status['stable_cards']=ready;(ROOT/'stable_free_event.json').write_text(json.dumps(status,indent=2)+'\n')
            print('STABLE FREE CARD',ready,'for120seconds.',flush=True)
            if '--recover036'not in sys.argv:return
            run=ROOT.parent/'runs/036'
            env={**os.environ,'DASO_REHOME_RUN':'036','DASO_REHOME_GPU':str(ready[0])}
            before=(run/'configs/sweep_formal_device.json').read_bytes()
            subprocess.run([sys.executable,ROOT/'code/rehome_llama_timing.py'],env=env,check=True)
            if (run/'configs/sweep_formal_device.json').read_bytes()==before:
                print('Recovery did not obtain an idle controller/card; keep observing.',flush=True)
                time.sleep(5);continue
            log=run/f'logs/controller_recovered_{time.time_ns()}.log'
            started=time.time()
            with log.open('w')as handle:
                child=subprocess.Popen([sys.executable,run/'code/sweep_driver.py'],stdout=handle,stderr=subprocess.STDOUT,env=env)
                (ROOT/'recovered036_controller.json').write_text(json.dumps({'pid':child.pid,'log':str(log),'started_unix':started,
                    'stability_window_seconds':120,'physical_gpu':ready[0],'configuration_and_quality_unchanged':True},indent=2)+'\n')
                print('Started own recovered036 controller',child.pid,'GPU',ready[0],str(log),flush=True)
                result=child.wait()
            print('Recovered036 controller exit',result,flush=True)
            if result:raise RuntimeError(str(log))
            return
        time.sleep(5)
    print('Observation window ended without sustained free card.',flush=True)


if __name__=='__main__':main()
