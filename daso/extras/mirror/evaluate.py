#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from extras.common.config import load_config, write_json
from extras.mirror.datasets import MirrorDataset, discover_genimage_zip, discover_labeled_folders
from extras.mirror.model import load_mirror_dense_detector


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float | int]:
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    for images, target in loader:
        logits, _, _ = model(images.to(device, non_blocking=True))
        probability = logits.softmax(-1)[:, 1]
        labels.extend(target.tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
        scores.extend(probability.cpu().tolist())
    y = np.asarray(labels)
    pred = np.asarray(predictions)
    if not len(y) or len(np.unique(y)) != 2:
        raise ValueError("MIRROR metrics require both real and fake samples")
    real_acc = float(np.mean(pred[y == 0] == 0))
    fake_acc = float(np.mean(pred[y == 1] == 1))
    return {"num_samples": len(labels), "accuracy": float(np.mean(pred == y)),
            "balanced_accuracy": (real_acc + fake_acc) / 2,
            "real_accuracy": real_acc, "fake_accuracy": fake_acc,
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MIRROR on Chameleon or GenImage.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("chameleon", "genimage"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["dataset"]
    if args.benchmark == "chameleon":
        splits = {"ALL": discover_labeled_folders(data["chameleon_path"], args.limit_per_class)}
        zip_path = None
    else:
        zip_path = data["genimage_zip"]
        splits = discover_genimage_zip(zip_path, args.limit_per_class)
    counts = {name: {"total": len(rows), "real": sum(row.label == 0 for row in rows),
                     "fake": sum(row.label == 1 for row in rows)} for name, rows in splits.items()}
    if args.discover_only:
        print(counts)
        return
    if args.output is None:
        raise ValueError("--output is required unless --discover-only is used")
    model_cfg = config["model"]
    model, _ = load_mirror_dense_detector(
        model_cfg["repository"], model_cfg["checkpoint"], model_cfg["memory"],
        model_cfg["backbone"], args.device)
    results = {}
    for name, records in splits.items():
        loader = DataLoader(MirrorDataset(records, zip_path), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
        results[name] = evaluate(model, loader, args.device)
    write_json(args.output, {"benchmark": args.benchmark, "counts": counts, "results": results})


if __name__ == "__main__":
    main()
