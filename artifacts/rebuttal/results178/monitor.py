"""Read-only process/clock samples; no GPU kernels or changes to other jobs."""
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
path=RUN/f'logs/device_monitor_{time.time_ns()}.jsonl'
with path.open('a',buffering=1) as handle:
 while True:
  state=json.loads((RUN/'results/worker.json').read_text())
  devices=[]
  for index in [0,3]:
   gpu=nvml.nvmlDeviceGetHandleByIndex(index);processes=[]
   for p in nvml.nvmlDeviceGetComputeRunningProcesses(gpu):
    command=Path(f'/proc/{p.pid}/cmdline')
    try:own=str(RUN).encode() in command.read_bytes()
    except (FileNotFoundError,PermissionError):own=None
    processes.append({'pid':p.pid,'used_bytes':p.usedGpuMemory,'belongs_to_run049':own})
   devices.append({'index':index,'processes':processes,'used_bytes':nvml.nvmlDeviceGetMemoryInfo(gpu).used,
    'sm_mhz':nvml.nvmlDeviceGetClockInfo(gpu,nvml.NVML_CLOCK_SM),
    'temperature_c':nvml.nvmlDeviceGetTemperature(gpu,nvml.NVML_TEMPERATURE_GPU),
    'gpu_utilization_percent':nvml.nvmlDeviceGetUtilizationRates(gpu).gpu})
  handle.write(json.dumps({'time_unix':time.time(),'stage':state,'devices':devices})+'\n')
  if state['status'] in ['complete','failed']:break
  time.sleep(2)
