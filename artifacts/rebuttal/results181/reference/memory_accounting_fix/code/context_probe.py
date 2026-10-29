"""Untimed CUDA context probe to calibrate NVML's unattributed-memory offset."""
from pathlib import Path
import argparse
import json
import os
import time
from telemetry import nvml,HANDLE

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
before=nvml.nvmlDeviceGetMemoryInfo(HANDLE).used
import torch
uuid=nvml.nvmlDeviceGetUUID(HANDLE)
assert str(torch.cuda.get_device_properties(0).uuid)==str(uuid).removeprefix('GPU-')
x=torch.empty(8*1024**2,dtype=torch.uint8,device='cuda');x.zero_();torch.cuda.synchronize();time.sleep(1)
samples=[]
for _ in range(5):
 used=nvml.nvmlDeviceGetMemoryInfo(HANDLE).used;processes={}
 for api in ['nvmlDeviceGetComputeRunningProcesses','nvmlDeviceGetGraphicsRunningProcesses']:
  for p in getattr(nvml,api)(HANDLE):processes[p.pid]=max(processes.get(p.pid,0),p.usedGpuMemory)
 own=processes.get(os.getpid(),0)
 samples.append({'time':time.time(),'used_bytes':used,'own_used_bytes':own,'own_visible':os.getpid() in processes,
  'unattributed_bytes':max(0,used-own),'foreign_pids':[p for p in processes if p!=os.getpid()]})
 time.sleep(1)
Path(args.output).write_text(json.dumps({'pid':os.getpid(),'uuid':uuid,'before_used_bytes':before,'samples':samples},indent=2)+'\n')
