"""Nine-hour, sustained-idle admission; never combine rounds across GPUs."""
from pathlib import Path
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from telemetry import nvml
from resource_state import memory

RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1]
D=json.loads((RUN/'configs/design.json').read_text());W=D['watch'];DEADLINE=W['deadline_unix']
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class ResourceInterruption(Exception):pass
class WindowEnded(Exception):pass
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def accepted(phase):return [read(p) for p in (RUN/'results/accepted').glob(phase+'_*.json')]
def grouped(phase,gpu,group):return [m for m in accepted(phase) if m['gpu']==gpu and m.get('group')==group]
def counts(phase):
 result={}
 for m in accepted(phase):result[m['gpu']]=result.get(m['gpu'],0)+1
 return result
def state(status='running',**kw):
 write(RUN/'results/worker.json',{'status':status,'deadline_unix':DEADLINE,'updated_unix':time.time(),
  'core_cohorts_by_gpu':counts('core_speed'),'long_cohorts_by_gpu':counts('long_speed'),
  'long_quality_jobs_by_gpu':counts('long_quality'),**kw})
def check_time():
 if time.time()>=DEADLINE:raise WindowEnded('Nine-hour window ended')
def invalidate(gpu,reason):
 path=RUN/'results/idle_invalidations.json';values=read(path) if path.exists() else {};values[str(gpu)]=time.time();write(path,values)
 with (RUN/'logs/idle_reset_events.jsonl').open('a') as handle:handle.write(json.dumps({'gpu':gpu,'time':values[str(gpu)],'reason':reason})+'\n')
def snapshot(gpu):
 h=nvml.nvmlDeviceGetHandleByIndex(gpu);mem=memory(h);processes={}
 for api in ['nvmlDeviceGetComputeRunningProcesses','nvmlDeviceGetGraphicsRunningProcesses']:
  for p in getattr(nvml,api)(h):
   value=p.usedGpuMemory if 0<=p.usedGpuMemory<=mem['total_bytes'] else 0
   processes[p.pid]=max(processes.get(p.pid,0),value)
 return {'time':time.time(),'gpu':gpu,'uuid':nvml.nvmlDeviceGetUUID(h),**mem,
  'utilization_percent':nvml.nvmlDeviceGetUtilizationRates(h).gpu,'processes':processes,
  'sm_mhz':nvml.nvmlDeviceGetClockInfo(h,nvml.NVML_CLOCK_SM),
  'temperature_c':nvml.nvmlDeviceGetTemperature(h,nvml.NVML_TEMPERATURE_GPU),'power_w':nvml.nvmlDeviceGetPowerUsage(h)/1000}
def bare(s):return s['used_bytes']<W['maximum_idle_used_bytes'] and s['utilization_percent']<=W['maximum_idle_utilization_percent'] and not s['processes']
def settle(gpu):
 good=0;observations=[];end=time.time()+18
 while time.time()<end:
  check_time();s=snapshot(gpu);observations.append(s);good=good+1 if bare(s) else 0
  if good>=D['post_exit_idle_samples']:return observations
  time.sleep(2)
 raise ResourceInterruption('Card did not return to unoccupied state after our process exited')
def choose(phase,preferred):
 check_time();p=RUN/'results/idle_status.json'
 if not p.exists():return None
 status=read(p)
 if status['status']!='monitoring' or time.time()-status['time_unix']>W['maximum_sampling_gap_seconds']:return None
 assert status['maximum_idle_used_bytes']==W['maximum_idle_used_bytes']
 assert status['memory_accounting']=='NVMLv2 total - free - reserved'
 reset_path=RUN/'results/idle_invalidations.json';resets=read(reset_path) if reset_path.exists() else {}
 eligible=[r for r in status['cards'] if r['eligible'] and r['idle_since_unix']>=resets.get(str(r['gpu']),0)]
 if preferred is not None and bare(snapshot(preferred)):
  return next((r for r in eligible if r['gpu']==preferred),None)
 progress=counts(phase);order={g:i for i,g in enumerate(W['preferred_cards'])}
 eligible.sort(key=lambda r:(-progress.get(r['gpu'],0),order[r['gpu']]))
 return eligible[0] if eligible else None
@contextlib.contextmanager
def lease(row,folder):
 root=BASE/'night151';phase=(root/'measurement.lock').open('a');card=(root/f"gpu_locks/gpu{row['gpu']}.lock").open('a')
 try:
  fcntl.flock(phase,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  card.close();phase.close();raise ResourceInterruption('Cooperative card/measurement lock busy')
 try:
  s=snapshot(row['gpu'])
  if not bare(s) or s['uuid']!=row['uuid']:raise ResourceInterruption('Card changed after idle admission')
  samples=[]
  for line in (RUN/'logs/idle_samples.jsonl').read_text().splitlines():
   d=json.loads(line)
   if d['time_unix']<row['idle_since_unix']:continue
   v=next(r for r in d['cards'] if r['gpu']==row['gpu'])
   samples.append({'time':d['time_unix'],**v})
  assert samples and samples[-1]['time']-samples[0]['time']>=W['required_idle_seconds']
  assert all(v['used_bytes']<W['maximum_idle_used_bytes'] and v['utilization_percent']<=5 and not v['visible_processes'] for v in samples)
  assert all(b['time']-a['time']<=W['maximum_sampling_gap_seconds'] for a,b in zip(samples,samples[1:]))
  write(folder/'admission.json',{'selected':row,'current':s,'samples':samples})
  yield
 finally:card.close();phase.close()
def env(gpu):return {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
 'TRANSFORMERS_OFFLINE':'1','TMPDIR':str(RUN/'tmp'),'TRITON_CACHE_DIR':str(RUN/'extensions/triton'),'CUDA_CACHE_PATH':str(RUN/'extensions/cuda')}
def command(gpu,*args):return ['numactl','--all',f'--physcpubind={gpu*8}-{gpu*8+7}',f'--membind={0 if gpu<4 else 1}',sys.executable,*map(str,args)]
def stop(process):
 if process.poll() is None:
  process.terminate()
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:process.kill();process.wait()
def calibrate(gpu,folder):
 path=folder/'calibration.json';argv=command(gpu,RUN/'code/context_probe.py','--output',path)
 with (folder/'calibration.log').open('w') as handle:
  p=subprocess.Popen(argv,env=env(gpu),stdout=handle,stderr=subprocess.STDOUT)
  try:
   while p.poll() is None:check_time();time.sleep(1)
  finally:stop(p)
 if p.returncode:raise ResourceInterruption('Context probe failed; see '+str(folder/'calibration.log'))
 data=read(path);v=data['samples']
 if data['before_used_bytes']>=W['maximum_idle_used_bytes']:raise ResourceInterruption('Card became occupied before context calibration')
 if not all(x['own_visible'] and not x['foreign_pids'] for x in v):raise ResourceInterruption('Context attribution unavailable or foreign process visible')
 gaps=[x['unattributed_bytes'] for x in v]
 if max(gaps)>1024**3 or max(gaps)-min(gaps)>64*1024**2:raise ResourceInterruption('Unstable/large context attribution offset')
 data['post_exit']=settle(gpu);data['allowed_unattributed_bytes']=max(gaps)+D['memory_gap_allowance_bytes'];data['artifact_path']=str(path.relative_to(RUN));write(path,data)
 return data
def run_job(gpu,phase,ri,pid,label,folder,calibration,version,hashes,group=None):
 check_time();full=f'{phase}_{group or "quality"}_g{gpu}_r{ri}_{label}_{time.time_ns()}'
 tier=1 if phase=='core_speed' else 2;mode='quality' if phase=='long_quality' else 'speed'
 argv=command(gpu,RUN/'code/experiment.py','--checkpoint',RUN/f'data/work/checkpoints/{pid}',
  '--policy',pid,'--mode',mode,'--tier',tier,'--round',ri,'--label',full)
 if group:argv.extend(['--group',group])
 log=RUN/f'logs/{full}.log';meta={'gpu':gpu,'phase':phase,'round':ri,'policy':pid,'label':label,'full_label':full,
  'command':argv,'code_version':version,'code_sha256':hashes,'group':group,'calibration':calibration['artifact_path']}
 write(log.with_suffix('.command.json'),meta);bad=0;reason=None;started=time.time();previous_usage=None
 with log.open('w') as handle,log.with_suffix('.monitor.jsonl').open('w') as monitor:
  process=subprocess.Popen(argv,env=env(gpu),stdout=handle,stderr=subprocess.STDOUT)
  try:
   while process.poll() is None:
    check_time();s=snapshot(gpu);own=s['processes'].get(process.pid,0);foreign=[k for k in s['processes'] if k!=process.pid]
    gap=max(0,s['used_bytes']-own)
    stable=own>0 and previous_usage is not None and abs(own-previous_usage[0])<=64*1024**2
    bad=bad+1 if stable and gap>calibration['allowed_unattributed_bytes'] else 0;previous_usage=(own,s['used_bytes'])
    s.update(expected_pid=process.pid,own_used_bytes=own,unattributed_bytes=gap,allocation_stable=stable,allowed_unattributed_bytes=calibration['allowed_unattributed_bytes'],foreign_pids=foreign)
    monitor.write(json.dumps(s)+'\n');monitor.flush();state(phase=phase,gpu=gpu,round=ri,policy=pid,job=full,elapsed_seconds=time.time()-started)
    if foreign or bad>=D['resource_confirmation_samples']:
     reason='External GPU process or sustained unattributed memory growth';break
    time.sleep(D['resource_sample_interval_seconds'])
  finally:stop(process)
 completion={'returncode':process.returncode,'seconds':time.time()-started,'resource_interruption':reason}
 write(log.with_suffix('.completion.json'),completion)
 if reason:raise ResourceInterruption(reason)
 if process.returncode:
  text=log.read_text(errors='replace')
  if any(s in text for s in ['CUDA out of memory','torch.OutOfMemoryError','Free memory on device (','cudaErrorMemoryAllocation']):raise ResourceInterruption('CUDA resource allocation failure')
  raise RuntimeError(str(log))
 post=settle(gpu);result=RUN/f'results/measurements/{full}.json';data=read(result)
 assert data['complete'] and int(data['gpu'])==gpu and data['gpu_uuid']==calibration['uuid']
 meta.update(result=str(result.relative_to(RUN)),result_sha256=sha(result),log=str(log.relative_to(RUN)),post_exit=post,completion=completion)
 write(folder/f'{label}.json',meta);return meta
def refresh():
 report=BASE/'results181/report053.py'
 if report.exists():subprocess.run([sys.executable,str(report)],check=True)
def winner(phase,target):
 complete=[]
 for gpu,count in counts(phase).items():
  if count==target:complete.append((max(m['finished_unix'] for m in accepted(phase) if m['gpu']==gpu),gpu))
 return min(complete)[1] if complete else None
def speed_phase(phase,preferred,version,hashes):
 groups=[name for name,spec in D['cohort_groups'].items() if spec['phase']==phase];target=3*len(groups)
 while winner(phase,target) is None:
  check_time();row=choose(phase,preferred)
  if row is None:
   state(phase=phase,waiting_for_sustained_idle=True,preferred_gpu=preferred);time.sleep(5);continue
  gpu=row['gpu'];lease_folder=RUN/f'results/leases/{phase}_g{gpu}_{time.time_ns()}';lease_folder.mkdir()
  try:
   with lease(row,lease_folder):
    calibration=calibrate(gpu,lease_folder)
    while counts(phase).get(gpu,0)<target:
     group=next(g for g in groups if len(grouped(phase,gpu,g))<3);ri=len(grouped(phase,gpu,group))
     folder=RUN/f'results/round_attempts/{phase}_{group}_g{gpu}_r{ri}_{time.time_ns()}';folder.mkdir()
     record={'phase':phase,'group':group,'gpu':gpu,'round':ri,'jobs':[],'started_unix':time.time(),'accepted':False,
      'admission':str((lease_folder/'admission.json').relative_to(RUN))}
     try:
      scenario=D['cohort_groups'][group]['scenario']
      order=[p for p in D['rounds'][ri] if p!='bf16' and D['policies'][p]['scenario']==scenario]
      for label in ['bf16_bracket_start',*order,'bf16_bracket_end']:
       pid='bf16' if label.startswith('bf16_bracket_') else label
       record['jobs'].append(run_job(gpu,phase,ri,pid,label,folder,calibration,version,hashes,group))
      record.update(accepted=True,finished_unix=time.time())
      write(RUN/f'results/accepted/{phase}_{group}_g{gpu}_r{ri}.json',record);preferred=gpu
     except Exception as error:
      record.update(interruption=str(error),finished_unix=time.time());raise
     finally:write(folder/'round.json',record)
     refresh()
  except ResourceInterruption as error:
   invalidate(gpu,str(error))
   write(lease_folder/'interruption.json',{'reason':str(error),'time_unix':time.time()})
  refresh()
  if winner(phase,target) is None:time.sleep(5)
 return winner(phase,target)
def quality_phase(preferred,version,hashes):
 policies=['bf16','pd_hawq_bal','pd_hawq_fast','pd_daso_bal','pd_daso_fast']
 while winner('long_quality',5) is None:
  check_time();row=choose('long_quality',preferred)
  if row is None:state(phase='long_quality',waiting_for_sustained_idle=True);time.sleep(5);continue
  gpu=row['gpu'];folder=RUN/f'results/round_attempts/long_quality_g{gpu}_{time.time_ns()}';folder.mkdir()
  try:
   with lease(row,folder):
    calibration=calibrate(gpu,folder)
    for pid in policies:
     marker=RUN/f'results/accepted/long_quality_g{gpu}_{pid}.json'
     if marker.exists():continue
     job=run_job(gpu,'long_quality',0,pid,pid,folder,calibration,version,hashes)
     write(marker,{'phase':'long_quality','gpu':gpu,'policy':pid,'finished_unix':time.time(),'jobs':[job],'admission':str((folder/'admission.json').relative_to(RUN))})
     preferred=gpu;refresh()
  except ResourceInterruption as error:
   invalidate(gpu,str(error));write(folder/'interruption.json',{'reason':str(error),'time_unix':time.time()});time.sleep(5)
 return winner('long_quality',5)
def main():
 with (RUN/'results/driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(RUN/'results/driver_pid.json',{'pid':os.getpid()})
  hashes=read(RUN/'configs/protocol_sha256.json')
  for name,digest in hashes.items():assert sha(RUN/'code'/name)==digest,name
  for path,digest in D['source_sha256'].items():assert sha(Path(path))==digest,path
  for spec in read(RUN/'reference/checkpoint_reuse.json').values():
   a=Path(spec['source']).stat();b=Path(spec['target']).stat()
   assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(spec['device'],spec['inode'],spec['bytes'])
   assert sha(Path(spec['target']).parent/'config.json')==spec['config_sha256']
  version=sha(RUN/'configs/protocol_sha256.json')[:16]
  for name in hashes:
   target=RUN/'code_versions'/version/name;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():shutil.copy2(RUN/'code'/name,target)
  core=speed_phase('core_speed',None,version,hashes);write(RUN/'results/core_complete.json',{'gpu':core,'finished_unix':time.time()})
  long=None;quality=None
  if DEADLINE-time.time()>=90*60:
   long=speed_phase('long_speed',core,version,hashes);write(RUN/'results/long_speed_complete.json',{'gpu':long,'finished_unix':time.time()})
  if long is not None and DEADLINE-time.time()>=75*60:
   quality=quality_phase(long,version,hashes);write(RUN/'results/long_quality_complete.json',{'gpu':quality,'finished_unix':time.time()})
  state(status='complete',phase='finished',core_gpu=core,long_gpu=long,long_quality_gpu=quality);refresh()

def interrupted(signum,frame):raise WindowEnded('Controller stop requested')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
if __name__=='__main__':
 try:main()
 except WindowEnded as error:state(status='window_ended',reason=str(error));refresh()
 except Exception as error:state(status='failed',error=repr(error));raise
