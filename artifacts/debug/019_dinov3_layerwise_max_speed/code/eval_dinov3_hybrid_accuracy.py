#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in (CODE_DIR, *CODE_DIR.parents) if (parent / "fake").is_dir() and (parent / "artifacts").is_dir()),
    CODE_DIR.parents[3],
)
for path in (CODE_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fake.data.dinov3_transforms import build_dinov3_lvd1689m_transform  # noqa: E402
from fake.data.imagenet_zip import DEFAULT_IMAGENET_ROOT, ImageNetZipDataset  # noqa: E402
from fake.evaluation.accuracy import evaluate_topk  # noqa: E402
from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config  # noqa: E402
from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config  # noqa: E402
from fake.models.dinov3 import DEFAULT_DINOV3_BACKBONE_PATH, DEFAULT_DINOV3_HEAD_PATH, model_input_dtype  # noqa: E402
from fake.models.dinov3_cutlass_hybrid import load_dinov3_vit7b16_cutlass_hybrid_classifier  # noqa: E402
from fake.utils.csv_io import append_csv_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing DINOv3 CUTLASS Hybrid classifier on ImageNet.")
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_IMAGENET_ROOT))
    parser.add_argument("--csv", default="val.csv")
    parser.add_argument("--zip", default="imagenet_val.zip")
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output", default="artifacts/debug/019_dinov3_layerwise_max_speed/hybrid_accuracy.csv")
    parser.add_argument("--hybrid-scheme", choices=["b16_manual", "b32_manual"], default="b32_manual")
    parser.add_argument("--no-prune", action="store_true", help="Require weights to already satisfy sparse patterns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    model, config, report = load_dinov3_vit7b16_cutlass_hybrid_classifier(
        backbone_path=args.backbone_path,
        head_path=args.head_path,
        device=device,
        hybrid_scheme=args.hybrid_scheme,
        sparse_nvfp4_config=CutlassSparseNVFP4Config(prune=not args.no_prune),
        sparse_bf16_config=CutlassSparseBF16Config(prune=not args.no_prune),
    )
    input_dtype = model_input_dtype(model)
    dataset = ImageNetZipDataset(
        args.dataset_root,
        args.csv,
        args.zip,
        transform=build_dinov3_lvd1689m_transform(args.resize_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    result = evaluate_topk(model, dataloader, device, input_dtype, args.log_interval)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "head": "dinov3_vit7b16_imagenet1k_linear_head",
        "method": "hybrid_cutlass",
        "hybrid_scheme": args.hybrid_scheme,
        "task": "imagenet_accuracy",
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device),
        "num_samples": result.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "resize_size": args.resize_size,
        "hidden_size": config.get("hidden_size", ""),
        "num_register_tokens": config.get("num_register_tokens", ""),
        "top1": f"{result.top1:.6f}",
        "top5": f"{result.top5:.6f}",
        "elapsed_sec": f"{result.elapsed_sec:.3f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "backbone_path": args.backbone_path,
        "head_path": args.head_path,
        "dataset_root": args.dataset_root,
        "csv": args.csv,
        "zip": args.zip,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **report.csv_fields(),
    }
    append_csv_row(args.output, list(row.keys()), row)
    print(
        "dinov3 hybrid accuracy done: "
        f"scheme={args.hybrid_scheme} top1={row['top1']} top5={row['top5']} "
        f"samples={result.num_samples} batch_size={args.batch_size} "
        f"sparse_nvfp4={report.sparse_nvfp4_module_count} sparse_bf16={report.sparse_bf16_module_count} "
        f"output={args.output}"
    )
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


if __name__ == "__main__":
    main()
