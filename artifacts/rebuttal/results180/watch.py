"""Audit the complete primary campaign, then both devices, without new GPU work."""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
RUN=BASE/'runs/051';OUT=BASE/'responses/reviewer2/qwen_quiet_speed_run051'
primary=False
while True:
 state=json.loads((RUN/'results/worker.json').read_text())
 audit=json.loads((OUT/'audit.json').read_text())
 if not primary and audit['per_device_summary']['3']['complete_rows']==35:
  subprocess.run([sys.executable,str(HERE/'verify_results.py'),'--gpu','3'],check=True)
  primary=True;print('Primary GPU3 complete and independently verified',flush=True)
 if state['status']=='complete':
  subprocess.run([sys.executable,str(HERE/'verify_results.py'),'--final'],check=True)
  print('Both devices complete and independently verified',flush=True);break
 if state['status']=='failed':raise RuntimeError(state)
 time.sleep(5)
