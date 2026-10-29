"""Resume CPU preparation after the pinned original Chat weights are ready."""
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
while not (RUN / 'results/source_model_complete.json').exists():
    time.sleep(10)
with (RUN / 'logs/cpu_prepare_after_restore.log').open('w') as handle:
    result = subprocess.run([sys.executable, RUN / 'code/prepare_model.py'],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=handle, stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError('See cpu_prepare_after_restore.log')
