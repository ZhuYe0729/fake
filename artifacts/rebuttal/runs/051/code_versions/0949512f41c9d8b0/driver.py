"""Serial, per-card timing campaigns; preserve every measurement attempt."""
from pathlib import Path
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

RUN=Path(__file__).resolve().parents[1]
BASE=RUN.parents[1]
DESIGN=json.loads((RUN/'configs/design.json').read_text())
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.tmp')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def state(status='running',**kw):
 write(RUN/'results/worker.json',{'status':status,'stage':'speed_only','primary_gpu':3,'validation_gpu':4,
  'completed_jobs':len(list((RUN/'results/completed').glob('*.json'))),'planned_jobs':54,'updated_unix':time.time(),**kw})

@contextlib.contextmanager
def lease(gpu):
 from telemetry import nvml
 root=BASE/'night151';device=nvml.nvmlDeviceGetHandleByIndex(gpu)
 while True:
  phase=(root/'measurement.lock').open('a')
  try:fcntl.flock(phase,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:phase.close();time.sleep(5);continue
  lock=(root/f'gpu_locks/gpu{gpu}.lock').open('a')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:lock.close();phase.close();time.sleep(5);continue
  if nvml.nvmlDeviceGetMemoryInfo(device).used>=512*1024**2 or nvml.nvmlDeviceGetUtilizationRates(device).gpu>5:
   lock.close();phase.close();time.sleep(5);continue
  uuid=nvml.nvmlDeviceGetUUID(device)
  p=RUN/f'configs/device_gpu{gpu}.json'
  if not p.exists():write(p,{'physical_gpu':gpu,'uuid':uuid,'name':nvml.nvmlDeviceGetName(device),
   'memory_bytes':nvml.nvmlDeviceGetMemoryInfo(device).total,'cpu_affinity':list(range(gpu*8,gpu*8+8)),
   'numa_node':0 if gpu<4 else 1,'frozen_unix':time.time()})
  assert read(p)['uuid']==uuid
  try:yield device,uuid
  finally:lock.close();phase.close()
  return

def job(gpu,ri,label,pid):
 from telemetry import nvml
 ident=f'g{gpu}_r{ri}_{label}';marker=RUN/f'results/completed/{ident}.json'
 if marker.exists():return
 hashes={str(p.relative_to(RUN/'code')):sha(p) for p in (RUN/'code').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
 for name in hashes:
  target=RUN/'code_versions'/version/name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():shutil.copy2(RUN/'code'/name,target)
 for retry in range(3):
  state(gpu=gpu,round=ri,policy=pid,job=ident,waiting_for_idle_card=True)
  with lease(gpu) as (device,uuid):
   attempt=str(time.time_ns());full=ident+'_'+attempt
   command=['numactl','--all',f'--physcpubind={gpu*8}-{gpu*8+7}',f'--membind={0 if gpu<4 else 1}',
    sys.executable,str(RUN/'code/experiment.py'),'--checkpoint',str(RUN/f'data/work/checkpoints/{pid}'),
    '--policy',pid,'--mode','speed','--tier','1','--round',str(ri),'--label',full]
   log=RUN/f'logs/{full}.log'
   write(log.with_suffix('.command.json'),{'command':command,'job':ident,'gpu':gpu,'gpu_uuid':uuid,'code_version':version,'code_sha256':hashes})
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
    'TMPDIR':str(RUN/'tmp'),'TRITON_CACHE_DIR':str(RUN/'extensions/triton'),'CUDA_CACHE_PATH':str(RUN/'extensions/cuda')}
   state(gpu=gpu,round=ri,policy=pid,job=ident,attempt=attempt,waiting_for_idle_card=False)
   started=time.time();foreign=set()
   with log.open('w') as handle,(RUN/f'logs/{full}.monitor.jsonl').open('w') as monitor:
    process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
    while process.poll() is None:
     processes=[{'pid':p.pid,'used_bytes':p.usedGpuMemory} for p in nvml.nvmlDeviceGetComputeRunningProcesses(device)]
     foreign.update(p['pid'] for p in processes if p['pid']!=process.pid)
     item={'time':time.time(),'expected_pid':process.pid,'gpu':gpu,'processes':processes,
      'sm_mhz':nvml.nvmlDeviceGetClockInfo(device,nvml.NVML_CLOCK_SM),'temperature_c':nvml.nvmlDeviceGetTemperature(device,nvml.NVML_TEMPERATURE_GPU),
      'power_w':nvml.nvmlDeviceGetPowerUsage(device)/1000}
     monitor.write(json.dumps(item)+'\n');monitor.flush();time.sleep(2)
   completion={'returncode':process.returncode,'seconds':time.time()-started,'foreign_pids':sorted(foreign)}
   write(log.with_suffix('.completion.json'),completion)
   result=RUN/f'results/measurements/{full}.json'
   if process.returncode==0 and not foreign:
    data=read(result);assert data['complete'] and data['gpu_uuid']==uuid and int(data['gpu'])==gpu
    write(marker,{'complete':True,'id':ident,'gpu':gpu,'gpu_uuid':uuid,'round':ri,'label':label,'policy':pid,
     'result':str(result.relative_to(RUN)),'result_sha256':sha(result),'log':str(log.relative_to(RUN)),'completion':completion})
    return
   output=log.read_text(errors='replace')
   resource=bool(foreign) or any(s in output for s in ['CUDA out of memory','torch.OutOfMemoryError','Free memory on device (','cudaErrorMemoryAllocation'])
   write(RUN/f'results/failed_attempts/{full}.json',{'job':ident,'resource_failure':resource,'completion':completion,'log':str(log)})
   if not resource:raise RuntimeError(str(log))
  time.sleep(5)
 raise RuntimeError('Three resource failures: '+ident)

def main():
 with (RUN/'results/driver.lock').open('a') as own:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,digest in read(RUN/'configs/protocol_sha256.json').items():assert sha(RUN/'code'/name)==digest,name
  for path,digest in DESIGN['source_sha256'].items():assert sha(Path(path))==digest,path
  for pid,spec in read(RUN/'reference/checkpoint_reuse.json').items():
   a=Path(spec['source']).stat();b=Path(spec['target']).stat()
   assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(spec['device'],spec['inode'],spec['bytes'])
   assert sha(Path(spec['target']).parent/'config.json')==spec['config_sha256']
  for gpu in DESIGN['cards']:
   for ri,order in enumerate(DESIGN['rounds']):
    for label in ['bf16_bracket_start',*[p for p in order if p!='bf16'],'bf16_bracket_end']:
     job(gpu,ri,label,'bf16' if label.startswith('bf16_bracket_') else label)
     report=BASE/'results180/report.py'
     if report.exists():subprocess.run([sys.executable,str(report)],check=True)
  assert len(list((RUN/'results/completed').glob('*.json')))==54
  state(status='complete',gpu=None,round=None,policy=None)

if __name__=='__main__':
 try:main()
 except Exception as error:state(status='failed',error=repr(error));raise
