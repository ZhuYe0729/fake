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

from fake.models.qwen3_5 import DEFAULT_QWEN3_5_VARIANT, QWEN3_5_VARIANTS, qwen3_5_model_path
from fake.models.qwen3_5_kernels import (
    QWEN3_5_REAL_KERNEL_METHODS,
    default_qwen3_5_kernel_checkpoint_path,
    prepare_qwen3_5_kernel_checkpoint_payload,
)


PREPARE_METHODS = tuple(method for method in QWEN3_5_REAL_KERNEL_METHODS if method != "dense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Qwen3.5 packed real-kernel checkpoint.")
    parser.add_argument("--method", choices=PREPARE_METHODS, required=True)
    parser.add_argument("--variant", choices=QWEN3_5_VARIANTS, default=DEFAULT_QWEN3_5_VARIANT)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device-map", default=None, help='Optional Transformers device_map, e.g. "auto".')
    parser.add_argument(
        "--max-memory",
        nargs="*",
        default=None,
        metavar="DEVICE:MEM",
        help='Optional max_memory entries, e.g. "0:30GiB" "1:30GiB".',
    )
    return parser.parse_args()


def _parse_max_memory(entries: list[str] | None) -> dict[int | str, str] | None:
    if not entries:
        return None
    result: dict[int | str, str] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid --max-memory entry {entry!r}; expected DEVICE:MEM")
        device, memory = entry.split(":", 1)
        result[int(device) if device.isdigit() else device] = memory
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to prepare Qwen3.5 real-kernel checkpoints.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model_path = args.model_path or str(qwen3_5_model_path(args.variant))
    output = Path(args.output or default_qwen3_5_kernel_checkpoint_path(args.variant, args.method))
    output.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    load_kwargs = dict(
        trust_remote_code=True,
        dtype=dtype,
        local_files_only=True,
    )
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
        max_memory = _parse_max_memory(args.max_memory)
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if args.device_map is None:
        model = model.to("cuda")
    model.eval()

    result = prepare_qwen3_5_kernel_checkpoint_payload(
        model,
        method=args.method,
        variant=args.variant,
        model_path=model_path,
        activation_dtype=dtype,
    )
    metadata = {
        **result.metadata,
        "checkpoint_path": str(output),
        "dtype": args.dtype,
    }
    torch.save({"state_dict": result.state_dict, "metadata": metadata}, output)
    with output.with_name("metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(
        "qwen3.5 kernel checkpoint prepared: "
        f"method={args.method} variant={args.variant} "
        f"modules={result.report.replaced_linear_count} skipped={result.report.skipped_linear_count} "
        f"output={output} bytes={output.stat().st_size}"
    )
    if result.report.skipped:
        print(f"skipped_modules={result.report.skipped[:10]}")


if __name__ == "__main__":
    main()
