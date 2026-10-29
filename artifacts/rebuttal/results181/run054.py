"""Parallel accuracy only, after every run053 speed group has completed."""
from pathlib import Path
import argparse
import contextlib
import fcntl
import os
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/054';SOURCE=BASE/'runs/053'
sys.path.insert(0,str(RUN/'code'))
import driver as d

parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
def state(status,**values):d.write(RUN/'results/worker.json',{'status':status,'updated_unix':time.time(),'deadline_unix':d.DEADLINE,'accuracy_only':True,**values})
def speed_ready():
 try:
  a=d.read(BASE/'responses/reviewer2/qwen_night_cohorts_run053/audit.json')
  v=d.read(BASE/'responses/reviewer2/qwen_night_cohorts_run053/long_speed_coverage_delivery_audit.json')
  return a['core_coverage_complete'] and a['long_coverage_complete'] and v['verified'] and v['long_coverage_complete']
 except (FileNotFoundError,ValueError,KeyError):return False
def verify_sources():
 hashes=d.read(RUN/'configs/protocol_sha256.json');version=d.sha(RUN/'configs/protocol_sha256.json')[:16]
 for name,digest in hashes.items():assert d.sha(RUN/'code'/name)==d.sha(SOURCE/'code'/name)==d.sha(RUN/'code_versions'/version/name)==digest
 for path,digest in d.D['source_sha256'].items():assert d.sha(Path(path))==digest
 for path,digest in d.read(RUN/'reference/input_sha256.json').items():assert d.sha(Path(path))==digest
 for spec in d.read(RUN/'reference/checkpoint_reuse.json').values():
  a=Path(spec['source']).stat();b=Path(spec['target']).stat();assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(spec['device'],spec['inode'],spec['bytes'])
  assert d.sha(Path(spec['target']).parent/'config.json')==spec['config_sha256']
 return version,hashes
def eligible():
 try:status=d.read(SOURCE/'results/idle_status.json')
 except (FileNotFoundError,ValueError):return []
 if status['status']!='monitoring' or time.time()-status['time_unix']>45:return []
 reset_path=SOURCE/'results/idle_invalidations.json';resets=d.read(reset_path) if reset_path.exists() else {}
 rows=[v for v in status['cards'] if v['eligible'] and v['idle_since_unix']>=resets.get(str(v['gpu']),0) and d.bare(d.snapshot(v['gpu']))]
 return sorted(rows,key=lambda v:(-v['continuous_idle_seconds'],v['gpu']))
def refresh():subprocess.run([sys.executable,str(BASE/'results181/report054.py')],check=True)
def batch(rows,version,hashes):
 roles=[(p,p) for p in d.D['batch_policies']]
 if len(rows)>=6:roles.append(('bf16_repeat','bf16'))
 samples=[__import__('json').loads(line) for line in (SOURCE/'logs/idle_samples.jsonl').read_text().splitlines()]
 children=[];record={'started_unix':time.time(),'deadline_unix':d.DEADLINE,'accuracy_only':True,'code_version':version,
  'controller':str(Path(__file__).resolve()),'controller_sha256':d.sha(Path(__file__)),'jobs':[]}
 d.write(RUN/'results/dispatch.json',record)
 try:
  for row,(role,policy) in zip(rows,roles):
   gpu=row['gpu'];current=d.snapshot(gpu);assert d.bare(current) and current['uuid']==row['uuid']
   history=[{'time':v['time_unix'],**next(c for c in v['cards'] if c['gpu']==gpu)} for v in samples if v['time_unix']>=row['idle_since_unix']]
   assert history[-1]['time']-history[0]['time']>=90
   assert all(v['used_bytes']<128*1024**2 and v['utilization_percent']<=5 and not v['visible_processes'] for v in history)
   assert all(b['time']-a['time']<=45 for a,b in zip(history,history[1:]))
   admission=RUN/f'reference/admissions/gpu{gpu}.json';d.write(admission,{'selected':row,'current':current,'samples':history,'source_monitor':str(SOURCE/'logs/idle_samples.jsonl')})
   label=f'quality_{role}_g{gpu}_{time.time_ns()}'
   command=d.command(gpu,RUN/'code/experiment.py','--checkpoint',RUN/f'data/work/checkpoints/{policy}',
    '--policy',policy,'--mode','quality','--tier',2,'--round',0,'--label',label)
   environment=d.env(gpu);environment.update(TRITON_CACHE_DIR=d.D['compiler_cache']['triton'],CUDA_CACHE_PATH=d.D['compiler_cache']['cuda'])
   log=RUN/f'logs/{label}.log';handle=log.open('w');started=time.time()
   process=subprocess.Popen(command,env=environment,stdout=handle,stderr=subprocess.STDOUT)
   job={'role':role,'policy':policy,'gpu':gpu,'uuid':row['uuid'],'label':label,'command':command,'code_sha256':hashes,
    'code_version':version,'admission':str(admission.relative_to(RUN)),'started_unix':started,'pid':process.pid,
    'log':str(log.relative_to(RUN)),'compiler_cache':d.D['compiler_cache']}
   d.write(log.with_suffix('.command.json'),job);record['jobs'].append(job);children.append((process,handle,job));d.write(RUN/'results/dispatch.json',record)
  finished=set()
  while any(p.poll() is None for p,_,_ in children):
   d.check_time()
   for process,handle,job in children:
    if process.poll() is None:
     sample=d.snapshot(job['gpu']);sample.update(expected_pid=process.pid,accuracy_only=True)
     with (RUN/job['log']).with_suffix('.monitor.jsonl').open('a') as monitor:monitor.write(__import__('json').dumps(sample)+'\n')
    elif job['role'] not in finished:
     finish(process,handle,job);finished.add(job['role'])
   state('running',jobs=[{**j,'returncode':p.poll(),'elapsed_seconds':time.time()-j['started_unix']} for p,_,j in children]);time.sleep(2)
  for process,handle,job in children:
   if job['role'] not in finished:finish(process,handle,job)
 finally:
  for process,handle,job in children:
   if process.poll() is None:d.stop(process)
   if not handle.closed:finish(process,handle,job)
  record['finished_unix']=time.time();d.write(RUN/'results/dispatch.json',record);refresh()
def finish(process,handle,job):
 handle.close();log=RUN/job['log'];completion={'returncode':process.returncode,'finished_unix':time.time(),'seconds':time.time()-job['started_unix'],'accuracy_only':True}
 d.write(log.with_suffix('.completion.json'),completion)
 result=RUN/f"results/measurements/{job['label']}.json"
 if process.returncode==0 and result.exists():
  value=d.read(result);assert value['complete'] and value['gpu_uuid']==job['uuid'] and value['policy']==job['policy']
  assert value['engine']==d.D['engine'] and value['policy_map_sha256']==d.D['policies'][job['policy']]['map_sha256']
  assert all(v['exact_equal'] for v in value['weight_audit'].values())
  marker={**job,'result':str(result.relative_to(RUN)),'result_sha256':d.sha(result),'completion':completion}
  d.write(RUN/f"results/completed/{job['role']}.json",marker)
def main():
 if args.check:
  if not speed_ready():raise SystemExit('Speed phase not complete; no GPU quality job launched')
  print('Speed gate ready');return
 version,hashes=verify_sources()
 d.write(RUN/'results/driver_pid.json',{'pid':os.getpid(),'entrypoint':str(Path(__file__).resolve())})
 while True:
  if d.DEADLINE-time.time()<d.D['minimum_remaining_seconds']:
   state('skipped',reason='Less than 180 seconds remain; no quality batch launched');refresh();return
  if not speed_ready():state('waiting_for_speed',minimum_remaining_seconds=180);time.sleep(5);continue
  previous=d.read(BASE/'responses/reviewer2/qwen_night_cohorts_run053/audit.json')
  if previous['long_quality_gpu'] is not None:state('skipped',reason='Long NLL already completed in run053');refresh();return
  rows=eligible()
  if len(rows)<5:state('waiting_for_five_idle_cards',eligible_gpus=[v['gpu'] for v in rows]);time.sleep(5);continue
  with contextlib.ExitStack() as stack:
   try:
    for path,mode in [(RUN/'results/driver.lock',fcntl.LOCK_EX),(SOURCE/'results/driver.lock',fcntl.LOCK_EX),(BASE/'night151/measurement.lock',fcntl.LOCK_SH)]:
     lock=stack.enter_context(path.open('a'));fcntl.flock(lock,mode|fcntl.LOCK_NB)
   except BlockingIOError:time.sleep(2);continue
   claimed=[]
   for row in rows:
    lock=(BASE/f"night151/gpu_locks/gpu{row['gpu']}.lock").open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();continue
    stack.callback(lock.close);claimed.append(row)
    if len(claimed)==6:break
   if len(claimed)<5:time.sleep(2);continue
   d.write(RUN/'results/driver_pid.json',{'pid':os.getpid(),'entrypoint':str(Path(__file__).resolve())})
   batch(claimed,version,hashes)
   count=sum((RUN/f'results/completed/{p}.json').exists() for p in d.D['batch_policies'])
   state('complete' if count==5 else 'partial',completed_main_policies=count);refresh();return

if __name__=='__main__':
 try:main()
 except d.WindowEnded as error:state('window_ended',reason=str(error));refresh()
 except Exception as error:state('failed',error=repr(error));refresh();raise
