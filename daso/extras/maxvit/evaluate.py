#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from extras.common.accuracy import evaluate_topk
from extras.common.config import load_config, write_json
from extras.common.imagenet_zip import ImageNetZipDataset
from extras.maxvit.model import load_maxvit_dense, model_input_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a dense MaxViT checkpoint on ImageNet ZIP.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg, data_cfg = config["model"], config["dataset"]
    model, runtime_cfg = load_maxvit_dense(
        model_cfg["path"], args.dtype, args.device, model_cfg.get("variant", "tiny"))
    dataset = ImageNetZipDataset(
        data_cfg["path"], data_cfg.get("csv_name", "val.csv"),
        data_cfg.get("zip_name", "imagenet_val.zip"), runtime_cfg)
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"))
    write_json(args.output, evaluate_topk(model, loader, args.device, model_input_dtype(model)))


if __name__ == "__main__":
    main()

