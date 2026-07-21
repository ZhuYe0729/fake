#!/usr/bin/env python3
"""Run the unchanged 058 discrete solver against the 061 warmed calibration."""
from __future__ import annotations
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE_SOLVER = ROOT / "artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/scripts/solve_pareto.py"


if __name__ == "__main__":
    runpy.run_path(str(SOURCE_SOLVER), run_name="__main__")
