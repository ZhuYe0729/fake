"""Quality-only workers claim idle GPUs and preserve every attempt."""
from pathlib import Path
import argparse
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
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_name(f'.{p.name}.{os.getpid()}.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


@contextlib.contextmanager
def lease():
 from telemetry import nvml
 root=BASE/'night151'
 while True:
  phase=(root/'measurement.lock').open('a')
  try:fcntl.flock(phase,fcntl.LOCK_SH|fcntl.LOCK_NB)
  except BlockingIOError:phase.close();time.sleep(5);continue
  claim=None
  for gpu in DESIGN['preferred_cards']:
   handle=nvml.nvmlDeviceGetHandleByIndex(gpu)
   if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:continue
   lock=(root/f'gpu_locks/gpu{gpu}.lock').open('a')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lock.close();continue
   if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:lock.close();continue
   claim=lock;uuid=nvml.nvmlDeviceGetUUID(handle);break
  if claim:
   try:yield gpu,uuid
   finally:claim.close();phase.close()
   return
  phase.close();time.sleep(5)


def run(job,gpu,uuid):
 attempt=str(time.time_ns());label=job['id']+'_'+attempt
 command=['numactl','--all',f'--physcpubind={gpu*8}-{gpu*8+7}',f'--membind={0 if gpu<4 else 1}',
  sys.executable,str(RUN/'code/quality_task.py'),'--job',job['id'],'--attempt',attempt]
 hashes={str(p.relative_to(RUN/'code')):sha(p) for p in (RUN/'code').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
 for name in hashes:
  p=RUN/'code_versions'/version/name;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():shutil.copy2(RUN/'code'/name,p)
 log=RUN/f'logs/{label}.log'
 write(log.with_suffix('.command.json'),{'command':command,'physical_gpu':gpu,'gpu_uuid':uuid,
  'code_version':version,'code_sha256':hashes,'job':job})
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'DASO_PHYSICAL_GPU':str(gpu),
  'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
  'TMPDIR':str(RUN/'tmp'),'HF_DATASETS_CACHE':str(RUN/'tmp/datasets'),
  'TRITON_CACHE_DIR':str(RUN/'extensions/triton'),'CUDA_CACHE_PATH':str(RUN/'extensions/cuda')}
 start=time.time()
 with log.open('w') as handle:result=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
 write(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.time()-start})
 if result.returncode:
  text=log.read_text(errors='replace')
  memory=any(s in text for s in ['CUDA out of memory','torch.OutOfMemoryError','Free memory on device (','cudaErrorMemoryAllocation'])
  write(RUN/f'results/failures/{label}.json',{'job':job,'log':str(log),'memory_failure':memory})
  if memory:return False
  raise RuntimeError(str(log))
 target=RUN/f"results/attempts/{job['id']}/{attempt}/result.json";value=read(target)
 assert value['complete'] and value['job']==job
 write(RUN/f"results/completed/{job['id']}.json",{'complete':True,'job':job,'result':str(target.relative_to(RUN)),
  'result_sha256':sha(target),'gpu':gpu,'gpu_uuid':uuid})
 return True


def worker(anchors=False):
 for job in DESIGN['jobs']:
  if anchors and job['id'] not in DESIGN['anchor_jobs']:continue
  path=RUN/f"results/locks/{job['id']}.lock";path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if (RUN/f"results/completed/{job['id']}.json").exists():continue
   for attempt in range(3):
    with lease() as (gpu,uuid):
     if run(job,gpu,uuid):break
    time.sleep(5)
   else:raise RuntimeError('Three resource failures: '+job['id'])


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--anchors-only',action='store_true');args=parser.parse_args()
 if args.worker:return worker(args.anchors_only)
 with (RUN/'results/driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,digest in read(RUN/'configs/protocol_sha256.json').items():assert sha(RUN/'code'/name)==digest,name
  for group in ['source_sha256','protected_sha256']:
   for path,digest in DESIGN[group].items():assert sha(Path(path))==digest,path
  for path,digest in read(RUN/'reference/likelihood_inputs.json').items():assert sha(Path(path))==digest,path
  selected=[j for j in DESIGN['jobs'] if not args.anchors_only or j['id'] in DESIGN['anchor_jobs']]
  children=[]
  for i in range(min(DESIGN['workers'],len(selected))):
   log=(RUN/f'logs/worker_{i}_{time.time_ns()}.log').open('w')
   command=[sys.executable,__file__,'--worker']+(['--anchors-only'] if args.anchors_only else [])
   children.append((subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT),log))
  while any(p.poll() is None for p,_ in children):
   complete=sum((RUN/f"results/completed/{j['id']}.json").exists() for j in selected)
   write(RUN/'results/worker.json',{'status':'running','stage':'quality_anchors' if args.anchors_only else 'quality_only',
    'complete_jobs':complete,'total_jobs':len(selected),'speed_enabled':False,'updated_unix':time.time()})
   time.sleep(5)
  assert all(p.returncode==0 for p,_ in children)
  for _,log in children:log.close()
  assert all((RUN/f"results/completed/{j['id']}.json").exists() for j in selected)
  write(RUN/'results/worker.json',{'status':'anchors_complete' if args.anchors_only else 'complete',
   'stage':'quality_only','complete_jobs':len(selected),'total_jobs':len(selected),'speed_enabled':False,'updated_unix':time.time()})


if __name__=='__main__':
 try:main()
 except Exception as error:
  write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)})
  if '--worker' not in sys.argv:write(RUN/'results/worker.json',{'status':'failed','stage':'quality_only','error':repr(error),'speed_enabled':False})
  raise
