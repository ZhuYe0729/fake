#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.marlin_nvfp4 import MarlinNVFP4Config, prepare_marlin_nvfp4_packed_model
from fake.models.mirror import (
    DEFAULT_MIRROR_BACKBONE_PATH,
    DEFAULT_MIRROR_MEMORY_PATH,
    DEFAULT_MIRROR_MODEL_PATH,
    load_mirror_dense_detector,
)
from fake.models.qwen3_5 import DEFAULT_QWEN3_5_VARIANT, QWEN3_5_VARIANTS, qwen3_5_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Marlin NVFP4 packed checkpoint.")
    parser.add_argument("--model", choices=["mirror", "qwen3_5"], required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--qwen-variant", choices=QWEN3_5_VARIANTS, default=DEFAULT_QWEN3_5_VARIANT)
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--mirror-model-path", default=str(DEFAULT_MIRROR_MODEL_PATH))
    parser.add_argument("--mirror-memory-path", default=str(DEFAULT_MIRROR_MEMORY_PATH))
    parser.add_argument("--mirror-backbone-path", default=str(DEFAULT_MIRROR_BACKBONE_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to pack Marlin NVFP4 checkpoint.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    output = Path(args.output or _default_output(args))
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "mirror":
        model, model_config = load_mirror_dense_detector(
            model_path=args.mirror_model_path,
            memory_path=args.mirror_memory_path,
            backbone_path=args.mirror_backbone_path,
            device=device,
            torch_dtype=dtype,
        )
        model_name = "mirror"
        source_metadata = {
            "model_path": args.mirror_model_path,
            "memory_path": args.mirror_memory_path,
            "backbone_path": args.mirror_backbone_path,
            **model_config,
        }
    else:
        from transformers import AutoModelForCausalLM

        model_path = args.qwen_model_path or str(qwen3_5_model_path(args.qwen_variant))
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            local_files_only=True,
        ).to(device)
        model.eval()
        model_name = "qwen3_5"
        source_metadata = {
            "model_path": model_path,
            "model_variant": args.qwen_variant,
        }

    metadata, report = prepare_marlin_nvfp4_packed_model(
        model,
        model_name,
        MarlinNVFP4Config(activation_dtype=dtype),
    )
    metadata.update(
        {
            "checkpoint_path": str(output),
            "source_metadata": source_metadata,
        }
    )
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
    }
    torch.save(payload, output)
    with output.with_name("metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(
        "marlin nvfp4 checkpoint prepared: "
        f"model={args.model} modules={report.replaced_linear_count} "
        f"skipped={report.skipped_linear_count} output={output} bytes={output.stat().st_size}"
    )
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


def _default_output(args: argparse.Namespace) -> str:
    if args.model == "mirror":
        return "artifacts/checkpoints/mirror/marlin_nvfp4/model.pt"
    variant_key = args.qwen_variant.lower().replace(".", "_")
    return f"artifacts/checkpoints/qwen3_5_{variant_key}/marlin_nvfp4/model.pt"


if __name__ == "__main__":
    main()
