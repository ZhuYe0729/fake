#!/usr/bin/env python3
"""Run the PMPD evaluator after selecting a physical CUDA device explicitly."""
from __future__ import annotations

import argparse
import os
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--script", required=True)
    args, remaining = parser.parse_known_args()
    # vLLM's automatic device config selects logical cuda:0. Constrain that
    # logical device before importing torch/CUDA so it maps to the requested
    # physical GPU even when the parent launcher has a different environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    torch.cuda.set_device(0)
    sys.argv = [args.script, *remaining]
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
