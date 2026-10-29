"""Resume036's complete third round on GPU4, keeping its valid first two rounds."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT.parent/'runs/036'
sys.path.insert(0,str(RUN/'code'))
from telemetry import nvml


def main():
    assert json.loads((RUN/'configs/sweep_formal_device.json').read_text())['physical_gpu']==4
    since=None;last=None
    while time.time()<1793148789:
        used=nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(4)).used
        free=used<512*1024**2;now=time.time()
        if free:since=now if since is None else since
        else:since=None
        if free!=last:print(time.strftime('%H:%M:%S',time.gmtime()),'GPU4 free',free,'usedMiB',used/1024**2,flush=True);last=free
        if since is not None and now-since>=120:
            with(RUN/'results/sweep_driver.lock').open('a')as own:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
            log=RUN/f'logs/controller_round2_resume_{time.time_ns()}.log'
            with log.open('w')as handle:
                child=subprocess.Popen([sys.executable,RUN/'code/sweep_driver.py'],stdout=handle,stderr=subprocess.STDOUT,env=os.environ)
                record={'started_unix':now,'pid':child.pid,'physical_gpu':4,'kept_rounds':[0,1],'repeated_round':2,
                    'log':str(log),'rule':'Same card, all incomplete round2 repeated, original three-round design unchanged.'}
                (ROOT/'round2_resume_controller.json').write_text(json.dumps(record,indent=2)+'\n')
                print('Resuming own036',record,flush=True);code=child.wait()
            if code:raise RuntimeError(str(log))
            print('036 same-card final round audited',flush=True);return
        time.sleep(5)
    print('Window ended while waiting forGPU4; previous complete rounds preserved.',flush=True)


if __name__=='__main__':main()
