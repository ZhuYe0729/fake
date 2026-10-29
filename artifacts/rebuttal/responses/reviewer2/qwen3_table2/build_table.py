"""Rebuild the uniform-runtime table from frozen run015 evidence."""
import runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
runpy.run_path(str(ROOT / "artifacts/rebuttal/runs/015/code/build_report.py"), run_name="__main__")
