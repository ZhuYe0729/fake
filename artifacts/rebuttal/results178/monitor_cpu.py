"""Read main-thread scheduling counters without changing the measured process."""
from pathlib import Path
import json
import os
import sys
import time

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/049'
sys.path.insert(0,str(RUN/'code'))
from telemetry import nvml

os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
path=RUN/f'logs/cpu_monitor_{time.time_ns()}.jsonl';previous={}
with path.open('a',buffering=1) as handle:
 while True:
  state=json.loads((RUN/'results/worker.json').read_text());now=time.time();samples=[]
  if state.get('stage')=='formal_speed':
   gpu=nvml.nvmlDeviceGetHandleByIndex(state['gpu'])
   for proc in nvml.nvmlDeviceGetComputeRunningProcesses(gpu):
    try:
     if str(RUN).encode() not in Path(f'/proc/{proc.pid}/cmdline').read_bytes():continue
     values=[int(v) for v in Path(f'/proc/{proc.pid}/schedstat').read_text().split()]
     sample={'pid':proc.pid,'cpu_runtime_ns':values[0],'runqueue_wait_ns':values[1],'timeslices':values[2]}
     if proc.pid in previous:
      old=previous[proc.pid];elapsed=now-old['time']
      sample['cpu_fraction']=(values[0]-old['values'][0])/1e9/elapsed
      sample['runqueue_fraction']=(values[1]-old['values'][1])/1e9/elapsed
     previous[proc.pid]={'time':now,'values':values};samples.append(sample)
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
  handle.write(json.dumps({'time_unix':now,'stage':state,'samples':samples})+'\n')
  if state['status'] in ['complete','failed']:break
  time.sleep(1)
