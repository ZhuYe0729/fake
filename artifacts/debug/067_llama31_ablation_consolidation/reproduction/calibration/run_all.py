#!/usr/bin/env python3
"""Run the offline quality calibration-size sensitivity experiment."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    cache = Path("/tmp/matplotlib-cache")
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", str(cache))
    for script in ("run_sensitivity.py", "build_report.py"):
        subprocess.run([sys.executable, str(SCRIPTS / script)], check=True, env=environment)


if __name__ == "__main__":
    main()
