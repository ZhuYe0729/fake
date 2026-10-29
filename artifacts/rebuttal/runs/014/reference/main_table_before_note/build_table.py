"""Rebuild the complete main table from the frozen run013 evidence."""
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
runpy.run_path(str(ROOT / "artifacts/rebuttal/runs/013/code/write_report.py"), run_name="__main__")
