#!/usr/bin/env python3
"""Run all uniform and solved policies serially on the designated speed GPU."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from common import RUN, runtime_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=os.environ.get("COSPAQ_SPEED_GPU", "0"))
    parser.add_argument("--selection", help="comma-separated labels; default is p00-p04 plus all point_* policies")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=100)
    args = parser.parse_args()
    policies: dict[str, Path] = {f"uniform_p{index:02d}": RUN / f"policies/prefill_decode/p{index:02d}.json" for index in range(5)}
    with (RUN / "pareto/predicted_points.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            policies[row["policy_id"]] = RUN / "pareto/policies" / f"{row['policy_id']}.json"
    selected = args.selection.split(",") if args.selection else list(policies)
    unknown = set(selected) - set(policies)
    if unknown:
        raise ValueError(f"unknown closure labels: {sorted(unknown)}")
    env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for label in selected:
        subprocess.run([sys.executable, str(Path(__file__).with_name("closure_policy.py")),
                        "--policy", str(policies[label]), "--label", label, "--gpu", str(args.gpu),
                        "--runs", str(args.runs), "--blocks", str(args.blocks)], check=True, env=env)


if __name__ == "__main__":
    main()
