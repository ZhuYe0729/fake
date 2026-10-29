"""Check accepted phase results and finish with a fresh independent audit."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'runs/052'
OUT=BASE/'responses/reviewer2/qwen_night_idle_run052';verified=set()
(HERE/'watch_pid.json').write_text(json.dumps({'pid':os.getpid()})+'\n')
while True:
 try:
  state=json.loads((RUN/'results/worker.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
 for phase,field in [('core_speed','core_primary_gpu'),('long_speed','long_primary_gpu'),('long_quality','long_quality_gpu')]:
  if phase not in verified and audit[field] is not None:
   subprocess.run([sys.executable,str(HERE/'verify_results.py'),'--phase',phase],check=True)
   verified.add(phase);print(phase,'complete and independently verified',flush=True)
 if state['status']=='window_ended' and time.time()<audit['deadline_unix'] and state.get('reason')=='Controller stop requested':
  time.sleep(5);continue
 if state['status']=='failed' and time.time()<audit['deadline_unix']:
  time.sleep(5);continue
 if state['status'] in ['complete','window_ended','failed']:
  with (RUN/'results/driver.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:time.sleep(2);continue
   subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
   if state['status']!='failed':subprocess.run([sys.executable,str(HERE/'verify_results.py'),'--final'],check=True)
   print('Window/controller finished:',state['status'],flush=True);break
 time.sleep(5)
