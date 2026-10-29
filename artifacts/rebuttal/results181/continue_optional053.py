"""Use remaining authorized time without changing any frozen benchmark code."""
from pathlib import Path
import argparse
import fcntl
import os
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['long_speed','long_quality'],required=True)
parser.add_argument('--preferred-gpu',type=int,choices=range(8))
args=parser.parse_args()
RUN=Path(__file__).resolve().parents[1]/'runs/053'
sys.path.insert(0,str(RUN/'code'))
import driver as d

def main():
 with (RUN/'results/driver.lock').open('a') as lock:
  while True:
   d.check_time()
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:time.sleep(1)
  core=d.winner('core_speed',3*sum(v['phase']=='core_speed' for v in d.D['cohort_groups'].values()))
  assert core is not None,'Complete core speed first'
  long=d.winner('long_speed',6);quality=d.winner('long_quality',5)
  if args.phase=='long_quality':assert long is not None,'Complete long speed first'
  minimum=35*60 if args.phase=='long_speed' else 25*60
  if d.DEADLINE-time.time()<minimum:raise SystemExit('Optional phase not started: insufficient remaining window')
  hashes=d.read(RUN/'configs/protocol_sha256.json');version=d.sha(RUN/'configs/protocol_sha256.json')[:16]
  for name,digest in hashes.items():assert d.sha(RUN/'code'/name)==d.sha(RUN/'code_versions'/version/name)==digest
  for path,digest in d.D['source_sha256'].items():assert d.sha(Path(path))==digest
  for spec in d.read(RUN/'reference/checkpoint_reuse.json').values():
   a=Path(spec['source']).stat();b=Path(spec['target']).stat()
   assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(spec['device'],spec['inode'],spec['bytes'])
   assert d.sha(Path(spec['target']).parent/'config.json')==spec['config_sha256']
  preferred=args.preferred_gpu if args.preferred_gpu is not None else long if long is not None else core
  assert preferred in d.W['preferred_cards']
  d.write(RUN/'results/optional_continuations'/f'{time.time_ns()}.json',{
   'phase':args.phase,'time_unix':time.time(),'deadline_unix':d.DEADLINE,'minimum_remaining_seconds':minimum,
   'preferred_gpu':preferred,'entrypoint':str(Path(__file__).resolve()),'entrypoint_sha256':d.sha(Path(__file__)),
   'code_version':version,'command':sys.argv,'reason':'Continue within original window after conservative main dispatch thresholds'})
  d.write(RUN/'results/driver_pid.json',{'pid':os.getpid(),'entrypoint':str(Path(__file__).resolve())})
  d.state(phase=args.phase,continuation=True)
  if args.phase=='long_speed':
   long=d.speed_phase('long_speed',preferred,version,hashes)
   d.write(RUN/'results/long_speed_complete.json',{'gpu':long,'finished_unix':time.time(),'continuation':True})
  if quality is None and long is not None and d.DEADLINE-time.time()>=25*60:
   quality=d.quality_phase(long,version,hashes)
   d.write(RUN/'results/long_quality_complete.json',{'gpu':quality,'finished_unix':time.time(),'continuation':True})
  d.state(status='complete',phase='finished',core_gpu=core,long_gpu=long,long_quality_gpu=quality,continuation=True);d.refresh()

if __name__=='__main__':
 try:main()
 except d.WindowEnded as error:d.state(status='window_ended',reason=str(error));d.refresh()
 except Exception as error:d.state(status='failed',error=repr(error));d.refresh();raise
