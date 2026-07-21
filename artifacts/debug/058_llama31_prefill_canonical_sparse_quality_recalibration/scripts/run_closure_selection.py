#!/usr/bin/env python3
"""Run a disk-safe serial measured closure spanning the solved prefill frontier."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

SELECTED = ("point_000", "point_003", "point_005", "point_007", "point_008", "point_009", "point_010", "point_011", "point_012", "point_014")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--gpu", type=int, default=1); args = parser.parse_args()
    runner = Path(__file__).with_name("run_closure_point.py")
    for policy in SELECTED:
        subprocess.run([sys.executable, str(runner), "--policy", policy, "--gpu", str(args.gpu)], check=True)
        print(f"completed {policy}", flush=True)
if __name__ == "__main__": main()
