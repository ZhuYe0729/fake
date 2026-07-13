#!/usr/bin/env python3
"""Export Llama2 checkpoints for the no-fuse vLLM benchmark architecture."""

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

METHODS = [
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "w4a16_ours",
    "w4a16_vllm_builtin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--cutlass-wrapper-path", default=DEFAULT_CUTLASS_WRAPPER)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prune", action="store_true")
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


def quantization_config(method: str) -> dict[str, object] | None:
    if method == "dense_bf16":
        return None
    if method == "dense_nvfp4":
        return {
            "quant_method": "nvfp4_mytest",
            "group_size": 16,
            "modules_to_not_convert": ["lm_head"],
        }
    if method == "sparse_bf16":
        return {
            "quant_method": "sparse_bf16_mytest",
            "backend": "cusparselt",
            "modules_to_not_convert": ["lm_head"],
        }
    if method == "sparse_nvfp4":
        return {
            "quant_method": "sparse_nvfp4_mytest",
            "modules_to_not_convert": ["lm_head"],
        }
    if method == "w4a16_ours":
        return {
            "quant_method": "marlin_nvfp4_mytest",
            "modules_to_not_convert": ["lm_head"],
        }
    if method == "w4a16_vllm_builtin":
        return {
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
    raise ValueError(f"unknown method: {method}")


def update_config(output_dir: Path, method: str) -> None:
    config_path = output_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    config["architectures"] = ["LlamaNoFuseForCausalLM"]
    config["torch_dtype"] = "bfloat16"
    qconfig = quantization_config(method)
    if qconfig is None:
        config.pop("quantization_config", None)
    else:
        config["quantization_config"] = qconfig
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


def iter_linear_weight_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        names.extend([
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight",
        ])
    return names


@torch.no_grad()
def quantize_dense_nvfp4(name: str,
                         weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import quantize_weight_bf16

    if weight.dim() != 2:
        raise ValueError(f"{name} must be a 2D linear weight")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n % 32 != 0 or k % 32 != 0:
        raise ValueError(f"{name} unsupported dense NVFP4 shape {n}x{k}")
    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = quantize_weight_bf16(cuda_weight, original_dtype=weight.dtype)
    result = {
        f"{name}.weight": q.packed_weight.view(n, k // 2).cpu().contiguous(),
        f"{name}.weight_scale": q.scale.cpu().contiguous(),
        f"{name}.weight_global_scale": q.global_scale.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def quantize_sparse_nvfp4(name: str, weight: torch.Tensor,
                          prune: bool) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import quantize_sparse_weight_bf16

    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = quantize_sparse_weight_bf16(cuda_weight,
                                    original_dtype=weight.dtype,
                                    prune=prune)
    result = {
        f"{name}.sparse_weight": q.sparse_weight.cpu().contiguous(),
        f"{name}.metadata": q.metadata.cpu().contiguous(),
        f"{name}.weight_scale": q.scale.cpu().contiguous(),
        f"{name}.weight_global_scale": q.global_scale.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def quantize_sparse_bf16(name: str, weight: torch.Tensor,
                         prune: bool) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import pack_sparse_bf16_weight

    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = pack_sparse_bf16_weight(cuda_weight,
                                original_dtype=weight.dtype,
                                prune=prune,
                                backend="cusparselt")
    result = {
        f"{name}.sparse_weight": q.sparse_weight.cpu().contiguous(),
        f"{name}.metadata": q.metadata.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def quantize_w4a16(name: str, weight: torch.Tensor,
                   variant: str) -> dict[str, torch.Tensor]:
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
        result = {
            f"{name}.weight_packed": q.packed_weight.cpu().contiguous(),
            f"{name}.weight_scale": q.logical_scale.cpu().contiguous(),
            f"{name}.weight_global_scale": torch.reciprocal(
                q.global_scale).reshape(1).cpu().contiguous(),
        }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


def quantize_method(method: str, name: str, weight: torch.Tensor,
                    prune: bool) -> dict[str, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be a 2D linear weight")
    if method == "dense_nvfp4":
        return quantize_dense_nvfp4(name, weight)
    if method == "sparse_bf16":
        return quantize_sparse_bf16(name, weight, prune)
    if method == "sparse_nvfp4":
        return quantize_sparse_nvfp4(name, weight, prune)
    if method == "w4a16_ours":
        return quantize_w4a16(name, weight, "ours")
    if method == "w4a16_vllm_builtin":
        return quantize_w4a16(name, weight, "vllm_builtin")
    raise ValueError(f"unknown quantized method: {method}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_dir} already exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if args.method != "dense_bf16":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for quantized export")
        major, minor = torch.cuda.get_device_capability()
        if major != 12:
            raise RuntimeError(
                f"SM120-class GPU is required, got capability {major}.{minor}")
        sys.path.insert(0, args.cutlass_wrapper_path)

    copy_model_assets(model_path, output_dir)
    update_config(output_dir, args.method)

    with (model_path / "config.json").open() as f:
        config = json.load(f)
    num_layers = int(config["num_hidden_layers"])
    weight_map = read_weight_map(model_path)
    reader = LazyTensorReader(model_path, weight_map)

    linear_sources = set(iter_linear_weight_names(num_layers))
    output_tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, dict[str, object]] = {}

    if args.method == "dense_bf16":
        for name in sorted(weight_map):
            output_tensors[name] = reader.get(name).contiguous()
    else:
        for idx, source_name in enumerate(sorted(linear_sources)):
            base_name = source_name.removesuffix(".weight")
            weight = reader.get(source_name)
            output_tensors.update(
                quantize_method(args.method, base_name, weight, args.prune))
            manifest[base_name] = {
                "source": source_name,
                "shape": list(weight.shape),
                "method": args.method,
                "prune": bool(args.prune),
            }
            del weight
            print(f"converted {idx + 1}/{len(linear_sources)} {base_name}",
                  flush=True)

        for name in sorted(weight_map):
            if name in linear_sources:
                continue
            output_tensors[name] = reader.get(name).contiguous()

    save_path = output_dir / "model.safetensors"
    save_file(output_tensors, save_path, metadata={"format": "pt"})
    with (output_dir / f"nofuse_{args.method}_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"saved {save_path} ({os.path.getsize(save_path) / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()
