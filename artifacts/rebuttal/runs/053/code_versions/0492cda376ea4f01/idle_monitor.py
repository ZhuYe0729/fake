"""Observe all GPUs until the user's deadline; never create a CUDA context."""
from pathlib import Path
import json
import os
import time
from telemetry import nvml
from resource_state import memory

RUN=Path(__file__).resolve().parents[1]
SPEC=json.loads((RUN/'configs/watch.json').read_text())
since={};uuids={};previous=None
def write(path,value):
 temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
write(RUN/'results/idle_monitor_pid.json',{'pid':os.getpid()})
with (RUN/'logs/idle_samples.jsonl').open('a') as log:
 while time.time()<SPEC['deadline_unix']:
  now=time.time();rows=[]
  reset_path=RUN/'results/idle_invalidations.json'
  resets=json.loads(reset_path.read_text()) if reset_path.exists() else {}
  if previous is not None and now-previous>SPEC['maximum_sampling_gap_seconds']:since.clear()
  for gpu in SPEC['preferred_cards']:
   h=nvml.nvmlDeviceGetHandleByIndex(gpu);uuid=nvml.nvmlDeviceGetUUID(h)
   if since.get(gpu,0)<resets.get(str(gpu),0):since.pop(gpu,None)
   if uuids.get(gpu)!=uuid:since.pop(gpu,None);uuids[gpu]=uuid
   mem=memory(h);used=mem['used_bytes'];util=nvml.nvmlDeviceGetUtilizationRates(h).gpu
   processes=[]
   for api in ['nvmlDeviceGetComputeRunningProcesses','nvmlDeviceGetGraphicsRunningProcesses']:
    processes.extend({'pid':p.pid,'bytes':p.usedGpuMemory,'api':api} for p in getattr(nvml,api)(h))
   idle=used<SPEC['maximum_idle_used_bytes'] and util<=SPEC['maximum_idle_utilization_percent'] and not processes
   if idle:since.setdefault(gpu,now)
   else:since.pop(gpu,None)
   rows.append({'gpu':gpu,'uuid':uuid,**mem,'utilization_percent':util,'visible_processes':processes,
    'idle':idle,'idle_since_unix':since.get(gpu),'continuous_idle_seconds':now-since[gpu] if gpu in since else 0,
    'eligible':gpu in since and now-since[gpu]>=SPEC['required_idle_seconds']})
  entry={'status':'monitoring','time_unix':now,'deadline_unix':SPEC['deadline_unix'],
   'memory_accounting':'NVMLv2 total - free - reserved',
   'maximum_idle_used_bytes':SPEC['maximum_idle_used_bytes'],'required_idle_seconds':SPEC['required_idle_seconds'],'cards':rows}
  log.write(json.dumps(entry)+'\n');log.flush();write(RUN/'results/idle_status.json',entry)
  previous=now;time.sleep(SPEC['sample_interval_seconds'])
write(RUN/'results/idle_status.json',{'status':'deadline_reached','time_unix':time.time(),'deadline_unix':SPEC['deadline_unix'],'cards':rows})
