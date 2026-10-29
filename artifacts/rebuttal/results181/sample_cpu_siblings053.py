"""Passive CPU/SMT diagnostic; never alter the benchmark or create CUDA work."""
from pathlib import Path
import hashlib
import json
import os
import signal
import time

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/053';OUT=BASE/'results181/reference'
deadline=json.loads((RUN/'configs/design.json').read_text())['watch']['deadline_unix']
prefix=OUT/f'cpu_siblings_{time.time_ns()}';allowed=set(os.sched_getaffinity(0));os.nice(10)
def cpu_list(value):
 result=[]
 for token in value.strip().split(','):
  bounds=[int(x) for x in token.split('-')];result.extend(range(bounds[0],bounds[-1]+1))
 return result
topology={int(p.name[3:]):cpu_list((p/'topology/thread_siblings_list').read_text()) for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*')}
node1=set(cpu_list(Path('/sys/devices/system/node/node1/cpulist').read_text()))
def cpu_times():
 result={}
 for line in Path('/proc/stat').read_text().splitlines():
  fields=line.split()
  if fields[0].startswith('cpu') and fields[0][3:].isdigit():result[int(fields[0][3:])]=[int(x) for x in fields[1:9]]
 return result
def stop(signum,frame):
 global stopped
 stopped=True
stopped=False;signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
os.sched_setaffinity(0,{max(allowed & node1)})
metadata={'pid':os.getpid(),'started_unix':time.time(),'deadline_unix':deadline,'diagnostic_only':True,'new_gpu_measurements':False,
 'sampling_interval_seconds':2,'topology':topology,'sampler_entrypoint':str(Path(__file__).resolve()),
 'sampler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
prefix.with_suffix('.meta.json').write_text(json.dumps(metadata,indent=2)+'\n')
(OUT/'cpu_siblings_active.json').write_text(json.dumps({'pid':os.getpid(),'prefix':str(prefix)})+'\n')
previous=cpu_times();previous_sched={};count=0
with prefix.with_suffix('.jsonl').open('w') as handle:
 while not stopped and time.time()<deadline:
  time.sleep(min(2,max(0,deadline-time.time())))
  current=cpu_times();busy={}
  for core,values in current.items():
   old=previous[core];total=sum(values)-sum(old);idle=values[3]+values[4]-old[3]-old[4]
   busy[core]=1-idle/total if total else 0
  previous=current;s=json.loads((RUN/'results/worker.json').read_text());job=s.get('job');pid=None;main=None
  if job:
   log=RUN/'logs'/f'{job}.monitor.jsonl'
   if log.exists():
    for line in log.read_text().splitlines()[-2:][::-1]:
     try:pid=json.loads(line)['expected_pid'];break
     except (json.JSONDecodeError,KeyError):continue
  if pid:
   try:
    process=Path(f'/proc/{pid}');fields=(process/'stat').read_text().rsplit(')',1)[1].split();core=int(fields[36])
    affinity=sorted(os.sched_getaffinity(pid));siblings=[v for v in topology[core] if v!=core]
    forbidden={v for bound in affinity for v in topology[bound]};sampler_core=max((allowed & node1)-forbidden)
    os.sched_setaffinity(0,{sampler_core})
    sched=[int(x) for x in (process/'schedstat').read_text().split()];delta=[a-b for a,b in zip(sched,previous_sched[pid])] if pid in previous_sched else None;previous_sched[pid]=sched
    main={'pid':pid,'cpu':core,'state':fields[0],'affinity':affinity,'siblings':siblings,
     'sibling_busy_fraction':{v:busy[v] for v in siblings},'sibling_in_our_affinity':any(v in affinity for v in siblings),
     'schedstat':sched,'schedstat_delta':delta,'cpu_khz':int(Path(f'/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq').read_text())}
   except (FileNotFoundError,ProcessLookupError):pass
  row={'time_unix':time.time(),'job':job,'phase':s.get('phase'),'gpu':s.get('gpu'),'main_thread':main,'cpu_busy_fraction':busy,'sampler_affinity':sorted(os.sched_getaffinity(0))}
  handle.write(json.dumps(row)+'\n');handle.flush();count+=1
prefix.with_suffix('.completion.json').write_text(json.dumps({'finished_unix':time.time(),'samples':count,'stopped':stopped,'diagnostic_only':True})+'\n')
print(json.dumps({'prefix':str(prefix),'samples':count,'diagnostic_only':True}),flush=True)
