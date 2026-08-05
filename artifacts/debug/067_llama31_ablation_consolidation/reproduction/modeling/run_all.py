#!/usr/bin/env python3
"""Run the offline 064 ablation twice and verify deterministic data outputs."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent


def run(name: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / name)], check=True)


def hashes() -> dict[str, str]:
    paths = sorted((OUT / "quality").glob("*.csv")) + sorted((OUT / "quality").glob("*.json"))
    paths += sorted((OUT / "speed").glob("*.csv")) + sorted((OUT / "speed").glob("*.json"))
    return {str(path.relative_to(OUT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main() -> None:
    run("run_quality_ablation.py")
    run("run_kernel_generalization_ablation.py")
    run("run_speed_ablation.py")
    first = hashes()
    run("run_quality_ablation.py")
    run("run_kernel_generalization_ablation.py")
    run("run_speed_ablation.py")
    second = hashes()
    if first != second:
        changed = sorted(set(first) | set(second))
        changed = [path for path in changed if first.get(path) != second.get(path)]
        raise RuntimeError(f"non-deterministic outputs: {changed}")
    run("build_report.py")
    run("build_detailed_report.py")
    result = {"deterministic": True, "checked_files": len(second), "sha256": second}
    (OUT / "report/reproducibility.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
