#!/usr/bin/env python3
"""Serial speed-only closure; requires NLL closure to have already completed."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from run_closure_selection import SELECTED
from scenario import EXP

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--policies", help="Comma-separated policy ids; defaults to the solved closure selection.")
    args = parser.parse_args()
    selected = tuple(args.policies.split(",")) if args.policies else SELECTED
    runner = Path(__file__).with_name("run_closure_point.py")
    for policy in selected:
        speed = EXP / "pareto/closure/speed" / policy / "runs"
        if all((speed / f"measured_{i}.json").exists() for i in range(5)): continue
        if not (EXP / "pareto/closure/nll" / f"{policy}.json").exists(): raise RuntimeError(f"missing NLL closure for {policy}")
        # The combined runner notices existing NLL and executes only speed.
        subprocess.run([sys.executable, str(runner), "--policy", policy, "--gpu", str(args.gpu)], check=True)
        print(f"completed speed {policy}", flush=True)
if __name__ == "__main__": main()
