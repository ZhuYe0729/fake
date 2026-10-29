"""Refresh the quality-only tables and run the final independent audit."""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'runs/050'
previous=-1;next_refresh=0
while True:
 state=json.loads((RUN/'results/worker.json').read_text())
 count=len(list((RUN/'results/completed').glob('*.json')))
 final=state['status']=='complete'
 if (count!=previous and time.time()>=next_refresh) or final:
  subprocess.run([sys.executable,str(HERE/'build_quality.py')],check=True)
  previous=count;next_refresh=time.time()+60
  print('quality-only progress',count,'/92',flush=True)
 if final:
  subprocess.run([sys.executable,str(HERE/'verify_quality.py'),'--final'],check=True)
  break
 if state['status']=='failed':raise RuntimeError(state)
 time.sleep(5)
