"""Use the final authorized window for NLL after all requested speed groups."""
from pathlib import Path
import fcntl
import os
import time
import coverage053 as coverage

d=coverage.d;RUN=coverage.RUN


def main():
 with (RUN/'results/driver.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit('Another run053 controller is active')
  d.check_time()
  core=coverage.completed_groups('core_speed');long=coverage.completed_groups('long_speed')
  assert len(core)==sum(v['phase']=='core_speed' for v in d.D['cohort_groups'].values())
  assert len(long)==sum(v['phase']=='long_speed' for v in d.D['cohort_groups'].values())
  if d.DEADLINE-time.time()<15*60:raise SystemExit('Less than 15 minutes remain; optional NLL not started')
  hashes=d.read(RUN/'configs/protocol_sha256.json');version=d.sha(RUN/'configs/protocol_sha256.json')[:16]
  for name,digest in hashes.items():assert d.sha(RUN/'code'/name)==d.sha(RUN/'code_versions'/version/name)==digest
  for path,digest in d.D['source_sha256'].items():assert d.sha(Path(path))==digest
  for spec in d.read(RUN/'reference/checkpoint_reuse.json').values():
   a=Path(spec['source']).stat();b=Path(spec['target']).stat()
   assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(spec['device'],spec['inode'],spec['bytes'])
   assert d.sha(Path(spec['target']).parent/'config.json')==spec['config_sha256']
  entrypoint=Path(__file__).resolve()
  d.write(RUN/'results/coverage_dispatches'/f'{time.time_ns()}.json',{'phase':'long_quality','time_unix':time.time(),
   'deadline_unix':d.DEADLINE,'minimum_remaining_seconds':15*60,'entrypoint':str(entrypoint),
   'entrypoint_sha256':d.sha(entrypoint),'code_version':version,'existing_core_groups':core,'existing_long_groups':long,
   'reason':'Continue optional NLL within original deadline after conservative 25-minute dispatch threshold'})
  d.write(RUN/'results/driver_pid.json',{'pid':os.getpid(),'entrypoint':str(entrypoint)})
  preferred=max(d.accepted('long_speed'),key=lambda m:m['finished_unix'])['gpu']
  d.state(phase='long_quality',continuation=True)
  quality=d.quality_phase(preferred,version,hashes)
  d.write(RUN/'results/long_quality_complete.json',{'gpu':quality,'finished_unix':time.time(),'continuation':True})
  d.state(status='complete',phase='finished',completion_scope='per_group_same_gpu_three_rounds',
   core_groups=core,long_groups=long,long_quality_gpu=quality,continuation=True);d.refresh()


if __name__=='__main__':
 try:main()
 except d.WindowEnded as error:d.state(status='window_ended',reason=str(error));d.refresh()
 except Exception as error:d.state(status='failed',error=repr(error));d.refresh();raise
