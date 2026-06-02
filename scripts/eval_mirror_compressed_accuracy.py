#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_mirror_dense_accuracy as mirror_eval
from fake.compression.checkpoint import checkpoint_csv_fields
from fake.compression.pipeline import SUPPORTED_METHODS
from fake.kernels.marlin_nvfp4 import load_marlin_nvfp4_checkpoint_into_model
from fake.models.mirror import (
    DEFAULT_MIRROR_BACKBONE_PATH,
    DEFAULT_MIRROR_MEMORY_PATH,
    DEFAULT_MIRROR_MODEL_PATH,
    load_mirror_compressed_detector,
    load_mirror_dense_detector,
)
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIRROR compressed detector on forensic benchmarks.")
    parser.add_argument("--method", choices=["dense", "marlin_nvfp4", *SUPPORTED_METHODS], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--benchmarks", nargs="+", default=["Chameleon", "GenImage"])
    parser.add_argument("--chameleon-root", default=str(mirror_eval.DEFAULT_CHAMELEON_ROOT))
    parser.add_argument("--genimage-root", default=str(mirror_eval.DEFAULT_GENIMAGE_ROOT))
    parser.add_argument("--genimage-zip", default=str(mirror_eval.DEFAULT_GENIMAGE_ZIP))
    parser.add_argument("--prefer-extracted-genimage", action="store_true")
    parser.add_argument("--model-path", default=str(DEFAULT_MIRROR_MODEL_PATH))
    parser.add_argument("--memory-path", default=str(DEFAULT_MIRROR_MEMORY_PATH))
    parser.add_argument("--backbone-path", default=str(DEFAULT_MIRROR_BACKBONE_PATH))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output", default="artifacts/results/mirror_compressed/accuracy.csv")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--discover-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = mirror_eval.discover_splits(args)
    if not splits:
        raise RuntimeError("No benchmark data was discovered.")

    print("Discovered datasets:")
    for split in splits:
        total, real, fake = mirror_eval.summarize_records(split.records)
        print(f"  {split.benchmark}/{split.dataset}: total={total} real={real} fake={fake}")
    if args.discover_only:
        return

    mirror_eval.import_runtime_deps()
    mirror_eval.seed_everything(0)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device(args.device)
    checkpoint_metadata: dict[str, object] = {}
    report_fields: dict[str, object] = {}
    if args.method == "dense":
        model, _ = load_mirror_dense_detector(args.model_path, args.memory_path, args.backbone_path, device=device)
    elif args.method == "marlin_nvfp4":
        checkpoint = args.checkpoint or "artifacts/checkpoints/mirror/marlin_nvfp4/model.pt"
        model, _ = load_mirror_dense_detector(
            args.model_path,
            args.memory_path,
            args.backbone_path,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        checkpoint_metadata, report = load_marlin_nvfp4_checkpoint_into_model(model, checkpoint, device=device)
        args.checkpoint = checkpoint
        report_fields = report.csv_fields()
        if report.skipped:
            print(f"skipped_modules={report.skipped[:10]}")
    else:
        checkpoint = args.checkpoint or f"artifacts/checkpoints/mirror/{args.method}/model.pt"
        model, _, checkpoint_metadata = load_mirror_compressed_detector(
            checkpoint,
            model_path=args.model_path,
            memory_path=args.memory_path,
            backbone_path=args.backbone_path,
            device=device,
        )
        args.checkpoint = checkpoint

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, object]] = []
    for split in splits:
        total, real, fake = mirror_eval.summarize_records(split.records)
        if total == 0 or real == 0 or fake == 0:
            print(f"[Skip] {split.benchmark}/{split.dataset}: total={total} real={real} fake={fake}")
            continue
        zip_path = args.genimage_zip if split.records and split.records[0].source == "zip" else None
        dataset = mirror_eval.MirrorDataset(split.records, zip_path=zip_path)
        dataloader = mirror_eval.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=mirror_eval.SequentialSampler(dataset),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=mirror_eval.collate_skip_invalid,
        )
        print(f"\n[{split.benchmark}/{split.dataset}] Evaluating {total} images ...")
        result = mirror_eval.evaluate(dataloader, model, device, args.use_amp, args.log_interval)
        rows.append(
            {
                "timestamp": timestamp,
                "model": "MIRROR-DINOv3-Huge",
                "method": args.method,
                "benchmark": split.benchmark,
                "dataset": split.dataset,
                "num_samples": result["num_samples"],
                "real_samples": result["real_samples"],
                "fake_samples": result["fake_samples"],
                "acc": f"{result['acc']:.6f}",
                "real_acc": f"{result['real_acc']:.6f}",
                "fake_acc": f"{result['fake_acc']:.6f}",
                "bal_acc": f"{result['bal_acc']:.6f}",
                "auc": f"{result['auc']:.6f}",
                "ap": f"{result['ap']:.6f}",
                "elapsed_sec": f"{result['elapsed_sec']:.3f}",
                "images_per_sec": f"{result['images_per_sec']:.3f}",
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": gpu_name,
                "use_amp": args.use_amp,
                "model_path": args.model_path,
                "memory_path": args.memory_path,
                "backbone_path": args.backbone_path,
                **checkpoint_csv_fields(checkpoint_metadata, args.checkpoint, args.method),
                **report_fields,
            }
        )

    if not rows:
        raise RuntimeError("No metric rows were produced.")
    _append_mean_rows(rows)
    fieldnames = list(rows[0].keys())
    for row in rows:
        append_csv_row(args.output, fieldnames, row)
    print(f"\nSaved results -> {args.output}")


def _append_mean_rows(rows: list[dict[str, object]]) -> None:
    by_benchmark: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_benchmark.setdefault(str(row["benchmark"]), []).append(row)
    for benchmark_rows in by_benchmark.values():
        if len(benchmark_rows) < 2:
            continue
        mean_row = dict(benchmark_rows[0])
        mean_row["dataset"] = "MEAN"
        mean_row["num_samples"] = sum(int(row["num_samples"]) for row in benchmark_rows)
        mean_row["real_samples"] = sum(int(row["real_samples"]) for row in benchmark_rows)
        mean_row["fake_samples"] = sum(int(row["fake_samples"]) for row in benchmark_rows)
        for metric in ["acc", "real_acc", "fake_acc", "bal_acc", "auc", "ap"]:
            mean_row[metric] = f"{mirror_eval.np.mean([float(row[metric]) for row in benchmark_rows]):.6f}"
        mean_row["elapsed_sec"] = f"{sum(float(row['elapsed_sec']) for row in benchmark_rows):.3f}"
        mean_row["images_per_sec"] = ""
        rows.append(mean_row)


if __name__ == "__main__":
    main()
