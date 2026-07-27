#!/usr/bin/env python3
"""Profile exact Llama3.1 prefill M=16384 and decode M=8 shapes."""
from __future__ import annotations

import argparse
import json
import os
import sys

from common import CUTLASS, PROTOCOL, RUN, command_output, write_json

sys.path[:0] = [str(CUTLASS), str(CUTLASS / "modeling")]
from modeling.kernel_predictor import (  # noqa: E402
    DEFAULT_KERNELS, profile_conversion_shapes, profile_shapes,
    update_models_from_measurements,
)

SHAPES = (
    (8 * 2048, 6144, 4096),
    (8 * 2048, 4096, 4096),
    (8 * 2048, 28672, 4096),
    (8 * 2048, 4096, 14336),
    (8, 6144, 4096),
    (8, 4096, 4096),
    (8, 28672, 4096),
    (8, 4096, 14336),
)
WEIGHT_SHAPES = tuple(dict.fromkeys((n, k) for _, n, k in SHAPES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("COSPAQ_SPEED_GPU", "0")))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = RUN / "kernel_profile"
    exact = root / "exact/targeted_profile.csv"
    conversion = root / "conversion/targeted_conversion_profile.csv"
    if args.force or not exact.exists():
        profile_shapes(SHAPES, kernels=DEFAULT_KERNELS, gpu=args.gpu, warmup=args.warmup,
                       iters=args.iters, output_dir=exact.parent, seed=0)
    if args.force or not conversion.exists():
        profile_conversion_shapes(WEIGHT_SHAPES, gpu=args.gpu, warmup=args.warmup,
                                  iters=args.iters, output_dir=conversion.parent, seed=0)
    result = update_models_from_measurements(
        [exact, conversion], input_root=root / "unused", output_root=root / "modeling",
        merge_existing=False, kernels=DEFAULT_KERNELS, neighbors=4)
    gpu = command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
    write_json(root / "metadata.json", {"protocol": PROTOCOL, "shapes": SHAPES,
               "weight_shapes": WEIGHT_SHAPES, "gpu_argument": args.gpu,
               "gpu_inventory": gpu, "updated_kernels": result.updated_kernels,
               "updated_conversions": result.updated_conversions})
    print(json.dumps({"modeling": str(root / "modeling"), "kernels": result.updated_kernels}, indent=2))


if __name__ == "__main__":
    main()
