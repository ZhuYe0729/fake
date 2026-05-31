#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_mirror_dense_accuracy as mirror_eval
from fake.compression.pipeline import (
    SUPPORTED_METHODS,
    CompressionConfig,
    compress_model,
    default_calib_batch_size,
    default_calib_samples,
    default_int4_group_size,
    default_nvfp4_group_size,
)
from fake.models.mirror import (
    DEFAULT_MIRROR_BACKBONE_PATH,
    DEFAULT_MIRROR_MEMORY_PATH,
    DEFAULT_MIRROR_MODEL_PATH,
    load_mirror_dense_detector,
    model_input_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MIRROR fake-quant/pruned compressed checkpoints.")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--benchmarks", nargs="+", default=["Chameleon", "GenImage"])
    parser.add_argument("--chameleon-root", default=str(mirror_eval.DEFAULT_CHAMELEON_ROOT))
    parser.add_argument("--genimage-root", default=str(mirror_eval.DEFAULT_GENIMAGE_ROOT))
    parser.add_argument("--genimage-zip", default=str(mirror_eval.DEFAULT_GENIMAGE_ZIP))
    parser.add_argument("--prefer-extracted-genimage", action="store_true")
    parser.add_argument("--model-path", default=str(DEFAULT_MIRROR_MODEL_PATH))
    parser.add_argument("--memory-path", default=str(DEFAULT_MIRROR_MEMORY_PATH))
    parser.add_argument("--backbone-path", default=str(DEFAULT_MIRROR_BACKBONE_PATH))
    parser.add_argument("--calib-samples", type=int, default=None)
    parser.add_argument("--calib-batch-size", type=int, default=None)
    parser.add_argument("--calib-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--nvfp4-group-size", type=int, default=None)
    parser.add_argument("--nvfp4-scale-precision", default="fp16")
    parser.add_argument("--nvfp4-scale-rule", choices=["static_6", "four_over_six_mse"], default=None)
    parser.add_argument("--nvfp4-scale-remap", default="none")
    parser.add_argument("--int4-group-size", type=int, default=None)
    parser.add_argument("--int4-scale-precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--sparsegpt-block-size", type=int, default=128)
    parser.add_argument("--sparsegpt-percdamp", type=float, default=0.01)
    parser.add_argument("--save-full-masks", action="store_true")
    parser.add_argument("--save-full-scales", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    mirror_eval.import_runtime_deps()
    mirror_eval.seed_everything(args.seed)
    device = torch.device("cuda")
    model, model_config = load_mirror_dense_detector(
        model_path=args.model_path,
        memory_path=args.memory_path,
        backbone_path=args.backbone_path,
        device=device,
    )
    input_dtype = model_input_dtype(model)
    calib_samples = args.calib_samples if args.calib_samples is not None else default_calib_samples("mirror")
    calib_batch_size = (
        args.calib_batch_size if args.calib_batch_size is not None else default_calib_batch_size("mirror")
    )
    output_dir = Path(args.output_dir or f"artifacts/checkpoints/mirror/{args.method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = mirror_eval.discover_splits(args)
    records = tuple(record for split in splits for record in split.records)
    if not records:
        raise RuntimeError("No MIRROR calibration images were discovered.")
    zip_path = args.genimage_zip if any(record.source == "zip" for record in records) else None
    dataset = mirror_eval.MirrorDataset(records, zip_path=zip_path)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=calib_batch_size,
        shuffle=args.calib_shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=mirror_eval.collate_skip_invalid,
        generator=generator,
    )

    config = CompressionConfig(
        model_name="mirror",
        method=args.method,
        calib_samples=calib_samples,
        sparsity=args.sparsity,
        nvfp4_group_size=args.nvfp4_group_size or default_nvfp4_group_size(args.method),
        nvfp4_scale_precision=args.nvfp4_scale_precision,
        nvfp4_scale_rule=args.nvfp4_scale_rule or _default_nvfp4_scale_rule(args.method),
        nvfp4_scale_remap=args.nvfp4_scale_remap,
        int4_group_size=args.int4_group_size or default_int4_group_size(args.method),
        int4_scale_precision=args.int4_scale_precision,
        sparsegpt_block_size=args.sparsegpt_block_size,
        sparsegpt_percdamp=args.sparsegpt_percdamp,
        save_full_masks=args.save_full_masks,
        save_full_scales=args.save_full_scales,
    )
    metadata, masks, scales = compress_model(
        model=model,
        dataloader=dataloader,
        device=device,
        input_dtype=input_dtype,
        config=config,
    )
    metadata.update(
        {
            "checkpoint_path": str(output_dir / "model.pt"),
            "masks_path": str(output_dir / "masks.pt"),
            "scales_path": str(output_dir / "scales.pt"),
            "calib_shuffle": args.calib_shuffle,
            "seed": args.seed,
            "calib_records": len(records),
            **model_config,
        }
    )

    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, output_dir / "model.pt")
    torch.save(masks, output_dir / "masks.pt")
    torch.save(scales, output_dir / "scales.pt")
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(
        "mirror compression done: "
        f"method={args.method} modules={metadata['compressed_modules']} "
        f"calib_samples={calib_samples} output={output_dir}"
    )


def _default_nvfp4_scale_rule(method: str) -> str:
    if method in ("nvfp4_4over6_unstructured_sparse", "nvfp4_4over6_semi_structured_sparse"):
        return "four_over_six_mse"
    return "static_6"


if __name__ == "__main__":
    main()
