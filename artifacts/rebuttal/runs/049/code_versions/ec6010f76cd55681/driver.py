"""Run fixed quality jobs, then quiet-card timing with all repetitions kept."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch import RUN,checkpoint,invoke,read,write
from telemetry import nvml

DESIGN=read(RUN/'configs/design.json')
LOCKS=RUN.parents[1]/'night151'


def state(stage,**values):write(RUN/'results/worker.json',{'status':'running','stage':stage,'updated_unix':time.time(),**values})


@contextlib.contextmanager
def lease(exclusive=False):
 while True:
  phase=(LOCKS/'measurement.lock').open('a')
  try:fcntl.flock(phase,(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB)
  except BlockingIOError:phase.close();time.sleep(5);continue
  frozen=RUN/'configs/formal_device.json'
  cards=[read(frozen)['physical_gpu']] if exclusive and frozen.exists() else DESIGN['cards']
  claim=None
  for gpu in cards:
   handle=nvml.nvmlDeviceGetHandleByIndex(gpu)
   if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:continue
   lock=(LOCKS/f'gpu_locks/gpu{gpu}.lock').open('a')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lock.close();continue
   if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:lock.close();continue
   claim=lock
   if exclusive and not frozen.exists():write(frozen,{'physical_gpu':gpu,'name':nvml.nvmlDeviceGetName(handle),
    'cpu_affinity':list(range(gpu*8,gpu*8+8)),'numa_node':0 if gpu<4 else 1,'frozen_unix':time.time()})
   break
  if claim:
   try:yield gpu
   finally:claim.close();phase.close()
   return
  phase.close();time.sleep(5)


def job(pid,tier,mode,gpu,round_index=0,anchor=None):
 label=f'{mode}_t{tier}_r{round_index}_{anchor or pid}'
 if (RUN/f'results/measurements/{label}.json').exists():return
 model=checkpoint(pid,gpu)
 invoke([sys.executable,RUN/'code/experiment.py','--checkpoint',model,'--policy',pid,'--tier',tier,
  '--mode',mode,'--round',round_index,'--label',label],label,gpu)


def quality_worker(tier):
 for pid in DESIGN['policies']:
  path=RUN/f'results/locks/quality_t{tier}_{pid}.lock';path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('a') as handle:
   try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if (RUN/f'results/measurements/quality_t{tier}_r0_{pid}.json').exists():continue
   with lease() as gpu:job(pid,tier,'quality',gpu)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--quality-worker',action='store_true');parser.add_argument('--tier',type=int,choices=[1,2],default=1)
 args=parser.parse_args()
 if args.quality_worker:return quality_worker(args.tier)
 with (RUN/'results/driver.lock').open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,digest in read(RUN/'configs/protocol_sha256.json').items():assert hashlib.sha256((RUN/'code'/name).read_bytes()).hexdigest()==digest,name
  for path,digest in DESIGN['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
  if not (RUN/'results/source_identity_check.json').exists():
   state('source_identity')
   for path,digest in read(RUN/'reference/source_model_fingerprint.json')['files_sha256'].items():
    with Path(path).open('rb') as source:assert hashlib.file_digest(source,'sha256').hexdigest()==digest,path
   write(RUN/'results/source_identity_check.json',{'verified':True,'time_unix':time.time()})
  state('quality',tier=args.tier)
  children=[]
  for i in range(2):
   log=(RUN/f'logs/quality_worker_t{args.tier}_{i}_{time.time_ns()}.log').open('w')
   children.append((subprocess.Popen([sys.executable,__file__,'--quality-worker','--tier',str(args.tier)],stdout=log,stderr=subprocess.STDOUT),log))
  while any(p.poll() is None for p,_ in children):
   state('quality',tier=args.tier,completed=len(list((RUN/'results/measurements').glob(f'quality_t{args.tier}_*.json'))));time.sleep(5)
  assert all(p.returncode==0 for p,_ in children)
  for _,log in children:log.close()
  assert len(list((RUN/'results/measurements').glob(f'quality_t{args.tier}_*.json')))==8
  subprocess.run([sys.executable,RUN/'code/report.py','--tier',str(args.tier),'--allow-partial'],check=True)
  for ri,order in enumerate(DESIGN['rounds']):
   labels=['bf16_bracket_start',*[p for p in order if p!='bf16'],'bf16_bracket_end']
   for index,label in enumerate(labels):
    pid='bf16' if label.startswith('bf16_bracket_') else label
    marker=RUN/f'results/measurements/speed_t{args.tier}_r{ri}_{label}.json'
    if marker.exists():continue
    state('waiting_formal_card',tier=args.tier,round=ri,index=index,total=len(labels),policy=pid)
    with lease(exclusive=True) as gpu:
     state('formal_speed',tier=args.tier,round=ri,index=index,total=len(labels),policy=pid,gpu=gpu)
     job(pid,args.tier,'speed',gpu,ri,label)
   subprocess.run([sys.executable,RUN/'code/report.py','--tier',str(args.tier),'--allow-partial'],check=True)
  subprocess.run([sys.executable,RUN/'code/report.py','--tier',str(args.tier)],check=True)
  write(RUN/f'results/tier{args.tier}_complete.json',{'complete':True,'rounds':len(DESIGN['rounds']),'time_unix':time.time()})
  write(RUN/'results/worker.json',{'status':'complete','tier':args.tier,'stage':'audited','updated_unix':time.time()})


if __name__=='__main__':
 try:main()
 except Exception as error:
  write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)});raise
