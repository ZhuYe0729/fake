"""Fill missing workload groups with frozen run053 measurement functions."""
from pathlib import Path
import fcntl
import os
import sys
import time

RUN=Path(__file__).resolve().parents[1]/'runs/053'
sys.path.insert(0,str(RUN/'code'))
import driver as d


def completed_groups(phase):
 result={}
 for name,spec in d.D['cohort_groups'].items():
  if spec['phase']!=phase:continue
  candidates=[]
  for gpu in d.W['preferred_cards']:
   rows=d.grouped(phase,gpu,name)
   if len(rows)==3 and {r['round'] for r in rows}=={0,1,2}:
    candidates.append((max(r['finished_unix'] for r in rows),gpu))
  if candidates:result[name]=min(candidates)[1]
 return result


def speed_coverage(phase,preferred,version,hashes,request):
 groups=[g for g,v in d.D['cohort_groups'].items() if v['phase']==phase]
 while len(completed_groups(phase))<len(groups):
  d.check_time();row=d.choose(phase,preferred)
  if row is None:
   d.state(phase=phase,waiting_for_sustained_idle=True,preferred_gpu=preferred,dispatch='missing_group_coverage');time.sleep(5);continue
  gpu=row['gpu'];lease_folder=RUN/f'results/leases/{phase}_g{gpu}_{time.time_ns()}';lease_folder.mkdir()
  try:
   with d.lease(row,lease_folder):
    calibration=d.calibrate(gpu,lease_folder)
    while len(completed_groups(phase))<len(groups):
     group=next(g for g in groups if g not in completed_groups(phase));ri=len(d.grouped(phase,gpu,group))
     assert ri in [0,1,2]
     folder=RUN/f'results/round_attempts/{phase}_{group}_g{gpu}_r{ri}_{time.time_ns()}';folder.mkdir()
     record={'phase':phase,'group':group,'gpu':gpu,'round':ri,'jobs':[],'started_unix':time.time(),'accepted':False,
      'admission':str((lease_folder/'admission.json').relative_to(RUN)),'dispatch_request':str(request.relative_to(RUN))}
     try:
      scenario=d.D['cohort_groups'][group]['scenario']
      order=[p for p in d.D['rounds'][ri] if p!='bf16' and d.D['policies'][p]['scenario']==scenario]
      for label in ['bf16_bracket_start',*order,'bf16_bracket_end']:
       pid='bf16' if label.startswith('bf16_bracket_') else label
       record['jobs'].append(d.run_job(gpu,phase,ri,pid,label,folder,calibration,version,hashes,group))
      record.update(accepted=True,finished_unix=time.time())
      marker=RUN/f'results/accepted/{phase}_{group}_g{gpu}_r{ri}.json';assert not marker.exists()
      d.write(marker,record);preferred=gpu
     except Exception as error:
      record.update(interruption=str(error),finished_unix=time.time());raise
     finally:d.write(folder/'round.json',record)
     d.refresh()
  except d.ResourceInterruption as error:
   d.invalidate(gpu,str(error));d.write(lease_folder/'interruption.json',{'reason':str(error),'time_unix':time.time()})
  d.refresh()
  if len(completed_groups(phase))<len(groups):time.sleep(5)
 return completed_groups(phase),preferred


def main():
 with (RUN/'results/driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d.check_time()
  hashes=d.read(RUN/'configs/protocol_sha256.json');version=d.sha(RUN/'configs/protocol_sha256.json')[:16]
  for name,digest in hashes.items():assert d.sha(RUN/'code'/name)==d.sha(RUN/'code_versions'/version/name)==digest
  for path,digest in d.D['source_sha256'].items():assert d.sha(Path(path))==digest
  for spec in d.read(RUN/'reference/checkpoint_reuse.json').values():
   a=Path(spec['source']).stat();b=Path(spec['target']).stat()
   assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(spec['device'],spec['inode'],spec['bytes'])
   assert d.sha(Path(spec['target']).parent/'config.json')==spec['config_sha256']
  entrypoint=Path(__file__).resolve();request=RUN/'results/coverage_dispatches'/f'{time.time_ns()}.json'
  d.write(request,{'time_unix':time.time(),'deadline_unix':d.DEADLINE,'entrypoint':str(entrypoint),
   'entrypoint_sha256':d.sha(entrypoint),'code_version':version,'rule':'First group in frozen order not complete for three rounds on any single GPU; first completion per group, independent of performance',
   'existing_core_groups':completed_groups('core_speed'),'existing_long_groups':completed_groups('long_speed')})
  d.write(RUN/'results/driver_pid.json',{'pid':os.getpid(),'entrypoint':str(entrypoint)})
  core,preferred=speed_coverage('core_speed',None,version,hashes,request)
  d.write(RUN/'results/core_coverage_complete.json',{'groups':core,'finished_unix':time.time(),'scope':'three rounds on one GPU per group'})
  long={};quality=d.winner('long_quality',5)
  if d.DEADLINE-time.time()>=35*60:
   long,preferred=speed_coverage('long_speed',preferred,version,hashes,request)
   d.write(RUN/'results/long_coverage_complete.json',{'groups':long,'finished_unix':time.time(),'scope':'three rounds on one GPU per group'})
  if long and quality is None and d.DEADLINE-time.time()>=25*60:
   quality=d.quality_phase(preferred,version,hashes)
   d.write(RUN/'results/long_quality_complete.json',{'gpu':quality,'finished_unix':time.time()})
  d.state(status='complete',phase='finished',completion_scope='per_group_same_gpu_three_rounds',core_groups=core,long_groups=long,long_quality_gpu=quality);d.refresh()


if __name__=='__main__':
 try:main()
 except d.WindowEnded as error:d.state(status='window_ended',reason=str(error));d.refresh()
 except Exception as error:d.state(status='failed',error=repr(error));d.refresh();raise
