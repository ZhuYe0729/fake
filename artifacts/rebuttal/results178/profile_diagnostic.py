"""After formal timing, trace one existing point; never use profiled times as results."""
from pathlib import Path
import fcntl
import hashlib
import json
import sys
import time

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/049'
STATUS=RUN/'results/profile_diagnostic.json'
def write(d):STATUS.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
write({'status':'waiting_for_core','scope':'One existing speed-oriented DASO point, four core cases; profiler timings excluded from reported results.'})
while not (RUN/'results/tier1_complete.json').exists():time.sleep(5)
sys.path.insert(0,str(RUN/'code'))
from launch import invoke
from telemetry import nvml
gpu=json.loads((RUN/'configs/formal_device.json').read_text())['physical_gpu']
locks=BASE/'night151';deadline=time.monotonic()+60
phase=(locks/'measurement.lock').open('a');card=(locks/f'gpu_locks/gpu{gpu}.lock').open('a')
acquired=False
while time.monotonic()<deadline:
 try:
  fcntl.flock(phase,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used<512*1024**2:
   acquired=True;break
 except BlockingIOError:pass
 fcntl.flock(card,fcntl.LOCK_UN);fcntl.flock(phase,fcntl.LOCK_UN);time.sleep(2)
if not acquired:
 write({'status':'not_run','reason':'No quiet fixed GPU within diagnostic wait limit; core results unaffected.'});raise SystemExit(0)
destination=RUN/'diagnostics';destination.mkdir(exist_ok=True)
prefix=destination/'daso_speed_trace';label='diagnostic_nsys_daso_speed'
argv=['timeout','240','nsys','profile','--trace=cuda,nvtx,osrt,python-gil','--sample=none','--cpuctxsw=none',
 '--cuda-graph-trace=graph','--stats=true','--output',prefix,sys.executable,RUN/'code/experiment.py',
 '--checkpoint',RUN/'data/work/checkpoints/pd_daso_fast','--policy','pd_daso_fast','--mode','speed','--tier','1','--round','0','--label',label]
write({'status':'running','gpu':gpu,'command':list(map(str,argv)),'excluded_from_formal_results':True,
 'experiment_sha256':hashlib.sha256((RUN/'code/experiment.py').read_bytes()).hexdigest()})
try:
 invoke(argv,label,gpu)
 write({'status':'complete','gpu':gpu,'trace':str(prefix)+'.nsys-rep','label':label,'excluded_from_formal_results':True})
except RuntimeError as error:
 write({'status':'failed_or_timed_out','log':str(error),'excluded_from_formal_results':True,'core_results_unchanged':True})
finally:card.close();phase.close()
