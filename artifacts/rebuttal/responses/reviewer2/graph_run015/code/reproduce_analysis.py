"""Rebuild analysis using the preserved run015 source tree and raw evidence."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[6]
SCRIPTS = ['analyze_hawq.py', 'plot_frontiers.py', 'build_report.py']
for script in SCRIPTS:
    subprocess.run([sys.executable, str(ROOT / 'artifacts/rebuttal/runs/015/code' / script)], check=True)
