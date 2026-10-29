#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from extras.common.accuracy import evaluate_topk
from extras.common.config import load_config, write_json
from extras.common.imagenet_zip import ImageNetZipDataset
from extras.dinov3.model import load_dinov3_vit7b16_dense_classifier, model_input_dtype
from extras.dinov3.transforms import build_dinov3_lvd1689m_transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a dense DINOv3 classifier on ImageNet ZIP.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg, data_cfg = config["model"], config["dataset"]
    model, _ = load_dinov3_vit7b16_dense_classifier(
        model_cfg["path"], model_cfg["head"], args.device)
    dataset = ImageNetZipDataset(
        data_cfg["path"], data_cfg.get("csv_name", "val.csv"),
        data_cfg.get("zip_name", "imagenet_val.zip"),
        transform=build_dinov3_lvd1689m_transform(int(model_cfg.get("image_size", 256))))
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"))
    write_json(args.output, evaluate_topk(model, loader, args.device, model_input_dtype(model)))


if __name__ == "__main__":
    main()
