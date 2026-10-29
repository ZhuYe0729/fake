#!/usr/bin/env python3
"""Score a merged PMPD JSONL without rerunning generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cnn_dm", "cnn_dm_1000", "dsum", "IWSLT"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bertscore-model", type=Path, required=True)
    parser.add_argument("--bertscore-num-layers", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.vllm_root))
    from pmpd_eval_kit import pmpd_eval

    metric_args = SimpleNamespace(
        dataset=args.dataset,
        split=args.split,
        bertscore_model=args.bertscore_model,
        bertscore_num_layers=args.bertscore_num_layers,
    )
    generated = pmpd_eval.compute_metrics(metric_args, args.answers)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generated.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
