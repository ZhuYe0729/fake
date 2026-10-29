#!/usr/bin/env python3
"""Export a phase-switch test checkpoint for artifact-local vLLM loading."""

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
    parser.add_argument("--cutlass-wrapper-path", default=DEFAULT_CUTLASS_WRAPPER)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_model_assets(src: Path, dst: Path) -> None:
    skip_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    skip_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    for item in src.iterdir():
        if item.name in skip_names:
            continue
        target = dst / item.name
        if item.is_dir():
            if not item.name.startswith("."):
                shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file() and item.suffix not in skip_suffixes:
            shutil.copy2(item, target)


def read_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            return json.load(f)["weight_map"]
    safetensors = sorted(model_path.glob("*.safetensors"))
    if len(safetensors) != 1:
        raise FileNotFoundError(
            "Expected model.safetensors.index.json or one safetensors file")
    with safe_open(safetensors[0], framework="pt", device="cpu") as f:
        return {key: safetensors[0].name for key in f.keys()}


class LazyTensorReader:

    def __init__(self, model_path: Path, weight_map: dict[str, str]) -> None:
        self.model_path = model_path
        self.weight_map = weight_map

    def get(self, name: str) -> torch.Tensor:
        with safe_open(self.model_path / self.weight_map[name],
                       framework="pt",
                       device="cpu") as f:
            return f.get_tensor(name)


def layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def update_config(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    config["quantization_config"] = {
        "quant_method": "phase_switch_mytest",
        "prefill_method": "dense_nvfp4",
        "decode_method": "w4a16_ours",
        "modules_to_not_convert": ["lm_head"],
    }
    config["torch_dtype"] = "bfloat16"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


@torch.no_grad()
def quantize_prefill_dense_nvfp4(name: str,
                                 weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import quantize_weight_bf16

    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n % 64 != 0 or k % 128 != 0:
        raise ValueError(f"{name} requires N%64==0 and K%128==0, "
                         f"got N={n}, K={k}")
    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = quantize_weight_bf16(cuda_weight, original_dtype=weight.dtype)
    result = {
        f"{name}.ps_prefill_weight":
        q.packed_weight.view(n, k // 2).cpu().contiguous(),
        f"{name}.ps_prefill_weight_scale": q.scale.cpu().contiguous(),
        f"{name}.ps_prefill_weight_global_scale":
        q.global_scale.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def quantize_decode_marlin_nvfp4(name: str,
                                 weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import pack_marlin_nvfp4_weight

    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n % 64 != 0 or k % 128 != 0:
        raise ValueError(f"{name} requires N%64==0 and K%128==0, "
                         f"got N={n}, K={k}")
    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = pack_marlin_nvfp4_weight(cuda_weight,
                                 activation_dtype=torch.bfloat16,
                                 original_dtype=weight.dtype)
    result = {
        f"{name}.ps_decode_packed_weight": q.packed_weight.cpu().contiguous(),
        f"{name}.ps_decode_weight_scale": q.weight_scale.cpu().contiguous(),
        f"{name}.ps_decode_weight_global_scale":
        q.global_scale.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


def convert_linear(name: str, weight: torch.Tensor,
                   output_tensors: dict[str, torch.Tensor],
                   manifest: dict[str, object]) -> None:
    output_tensors.update(quantize_prefill_dense_nvfp4(name, weight))
    output_tensors.update(quantize_decode_marlin_nvfp4(name, weight))
    manifest[name] = {
        "shape": list(weight.shape),
        "prefill_method": "dense_nvfp4",
        "decode_method": "w4a16_ours",
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for phase-switch export")
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        raise RuntimeError(
            f"SM120-class GPU is required, got capability {major}.{minor}")
    sys.path.insert(0, args.cutlass_wrapper_path)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_dir} exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    copy_model_assets(model_path, output_dir)
    update_config(output_dir)

    with (model_path / "config.json").open() as f:
        config = json.load(f)
    num_layers = int(config["num_hidden_layers"])
    weight_map = read_weight_map(model_path)
    reader = LazyTensorReader(model_path, weight_map)

    linear_sources: set[str] = set()
    output_tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, object] = {}

    for layer_idx in range(num_layers):
        prefix = layer_prefix(layer_idx)
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
        convert_linear(f"{prefix}.self_attn.qkv_proj", fused_qkv,
                       output_tensors, manifest)
        del fused_qkv

        fused_gate_up = torch.cat([reader.get(name) for name in gate_up_names],
                                  dim=0)
        convert_linear(f"{prefix}.mlp.gate_up_proj", fused_gate_up,
                       output_tensors, manifest)
        del fused_gate_up

        for suffix in ("self_attn.o_proj", "mlp.down_proj"):
            source_name = f"{prefix}.{suffix}.weight"
            linear_sources.add(source_name)
            weight = reader.get(source_name)
            convert_linear(f"{prefix}.{suffix}", weight, output_tensors,
                           manifest)
            del weight

        print(f"converted layer {layer_idx + 1}/{num_layers}", flush=True)

    for name in sorted(weight_map):
        if name in linear_sources:
            continue
        output_tensors[name] = reader.get(name).contiguous()

    save_path = output_dir / "model.safetensors"
    save_file(output_tensors, save_path, metadata={"format": "pt"})
    with (output_dir / "phase_switch_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    total_bytes = os.path.getsize(save_path)
    print(f"saved {save_path} ({total_bytes / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()
