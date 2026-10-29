"""Wait without a lease, then isolate the CPU analysis from formal timing."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
while not(root.parent/'runs/041/results/final_audit.json').exists():time.sleep(20)
with(root.parent/'night151/measurement.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_SH)
    start=time.time();result=subprocess.run([sys.executable,root/'code/latency_gate_analysis.py'])
    (root/'latency_validation/action.json').write_text(json.dumps({'returncode':result.returncode,
        'start_unix':start,'elapsed_seconds':time.time()-start},indent=2)+'\n')
    raise SystemExit(result.returncode)
