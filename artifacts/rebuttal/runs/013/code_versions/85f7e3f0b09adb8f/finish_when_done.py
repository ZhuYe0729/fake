"""Publish the requested table only after every frozen evaluation completes."""
import json
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
while True:
    state = json.loads((RUN / 'results/worker.json').read_text())
    if state['status'] == 'complete':
        break
    if state['status'] == 'failed':
        raise RuntimeError(state)
    time.sleep(10)
for script in ['write_report.py', 'audit_package.py']:
    subprocess.run([sys.executable, RUN / 'code' / script], check=True)
