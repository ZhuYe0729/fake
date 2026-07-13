#!/usr/bin/env python3
"""Export Llama2 fused checkpoints for W4A16 NVFP4 benchmark paths."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_CUTLASS_WRAPPER = (
    "/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/"
    "cutlass_wrapper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant",
                        choices=["ours", "vllm_builtin"],
                        required=True)
    parser.add_argument("--cutlass-wrapper-path", default=DEFAULT_CUTLASS_WRAPPER)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_model_assets(src: Path, dst: Path) -> None:
    skip_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    skip_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    for item in src.iterdir():
        if item.name in skip_names:
            continue
        if item.is_file() and item.suffix in skip_suffixes:
            continue
        target = dst / item.name
        if item.is_dir():
            if item.name.startswith("."):
                continue
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)


def update_config(output_dir: Path, variant: str) -> None:
    config_path = output_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)

    if variant == "ours":
        config["quantization_config"] = {
            "quant_method": "marlin_nvfp4_mytest",
            "modules_to_not_convert": ["lm_head"],
        }
    else:
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "tensor_group",
                        "group_size": 16,
                        "dynamic": False,
                    },
                },
            },
        }
    config["torch_dtype"] = "bfloat16"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def read_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        safetensors = sorted(model_path.glob("*.safetensors"))
        if len(safetensors) != 1:
            raise FileNotFoundError(
                "Expected model.safetensors.index.json or one safetensors file")
        with safe_open(safetensors[0], framework="pt", device="cpu") as f:
            return {key: safetensors[0].name for key in f.keys()}
    with index_path.open() as f:
        return json.load(f)["weight_map"]


class LazyTensorReader:

    def __init__(self, model_path: Path, weight_map: dict[str, str]) -> None:
        self.model_path = model_path
        self.weight_map = weight_map

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        with safe_open(self.model_path / filename, framework="pt",
                       device="cpu") as f:
            return f.get_tensor(name)


@torch.no_grad()
def quantize_weight(name: str, weight: torch.Tensor,
                    variant: str,
                    global_scale_names: list[str] | None = None
                    ) -> dict[str, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be a 2D linear weight")
    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()

    if variant == "ours":
        from cutlass_wrapper import pack_marlin_nvfp4_weight

        q = pack_marlin_nvfp4_weight(cuda_weight,
                                     activation_dtype=torch.bfloat16,
                                     original_dtype=weight.dtype)
        result = {
            f"{name}.packed_weight": q.packed_weight.cpu().contiguous(),
            f"{name}.weight_scale": q.weight_scale.cpu().contiguous(),
            f"{name}.weight_global_scale": q.global_scale.cpu().contiguous(),
        }
    else:
        from cutlass_wrapper import quantize_nvfp4_canonical_weight

        q = quantize_nvfp4_canonical_weight(cuda_weight,
                                            original_dtype=weight.dtype)
        inverse_global_scale = torch.reciprocal(q.global_scale).reshape(1)
        result = {
            f"{name}.weight_packed": q.packed_weight.cpu().contiguous(),
            f"{name}.weight_scale": q.logical_scale.cpu().contiguous(),
        }
        scale_names = global_scale_names or [f"{name}.weight_global_scale"]
        for scale_name in scale_names:
            result[scale_name] = inverse_global_scale.cpu().contiguous()

    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


def layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for W4A16 export")
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        raise RuntimeError(
            f"SM120-class GPU is required, got capability {major}.{minor}")

    sys.path.insert(0, args.cutlass_wrapper_path)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_dir} already exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copy_model_assets(model_path, output_dir)
    update_config(output_dir, args.variant)

    with (model_path / "config.json").open() as f:
        config = json.load(f)
    num_layers = int(config["num_hidden_layers"])

    weight_map = read_weight_map(model_path)
    reader = LazyTensorReader(model_path, weight_map)
    linear_sources: set[str] = set()
    output_tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, dict[str, object]] = {}

    def convert(base_name: str,
                source: str,
                weight: torch.Tensor,
                global_scale_names: list[str] | None = None) -> None:
        output_tensors.update(
            quantize_weight(base_name, weight, args.variant,
                            global_scale_names))
        manifest[base_name] = {
            "source": source,
            "shape": list(weight.shape),
            "variant": args.variant,
            "global_scale_names": global_scale_names
            or [f"{base_name}.weight_global_scale"],
        }

    for i in range(num_layers):
        prefix = layer_prefix(i)
        qkv_names = [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ]
        gate_up_names = [
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
        ]
        for source_name in qkv_names + gate_up_names:
            linear_sources.add(source_name)

        fused_qkv = torch.cat([reader.get(name) for name in qkv_names], dim=0)
        convert(f"{prefix}.self_attn.qkv_proj", "q_proj+k_proj+v_proj",
                fused_qkv, [
                    f"{name.removesuffix('.weight')}.weight_global_scale"
                    for name in qkv_names
                ] if args.variant == "vllm_builtin" else None)
        del fused_qkv

        fused_gate_up = torch.cat([reader.get(name) for name in gate_up_names],
                                  dim=0)
        convert(f"{prefix}.mlp.gate_up_proj", "gate_proj+up_proj",
                fused_gate_up, [
                    f"{name.removesuffix('.weight')}.weight_global_scale"
                    for name in gate_up_names
                ] if args.variant == "vllm_builtin" else None)
        del fused_gate_up

        for suffix in ("self_attn.o_proj", "mlp.down_proj"):
            source_name = f"{prefix}.{suffix}.weight"
            linear_sources.add(source_name)
            weight = reader.get(source_name)
            convert(f"{prefix}.{suffix}", source_name, weight)
            del weight

        print(f"converted layer {i + 1}/{num_layers}", flush=True)

    for name in sorted(weight_map):
        if name in linear_sources:
            continue
        output_tensors[name] = reader.get(name).contiguous()

    save_path = output_dir / "model.safetensors"
    save_file(output_tensors, save_path, metadata={"format": "pt"})
    with (output_dir / f"w4a16_{args.variant}_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"saved {save_path} ({os.path.getsize(save_path) / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()
