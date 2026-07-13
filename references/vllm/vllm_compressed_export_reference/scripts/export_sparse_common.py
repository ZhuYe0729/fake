#!/usr/bin/env python3
"""Shared exporter for local sparse vLLM integration methods."""

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


def make_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutlass-wrapper-path", default=DEFAULT_CUTLASS_WRAPPER)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.set_defaults(method=method)
    return parser


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


def update_config(output_dir: Path, method: str) -> None:
    config_path = output_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    qconfig: dict[str, object] = {
        "quant_method": method,
        "modules_to_not_convert": ["lm_head"],
    }
    if method == "sparse_bf16_mytest":
        qconfig["backend"] = "cusparselt"
    config["quantization_config"] = qconfig
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
def quantize_sparse_nvfp4(name: str, weight: torch.Tensor,
                          prune: bool) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import quantize_sparse_weight_bf16

    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = quantize_sparse_weight_bf16(cuda_weight,
                                    original_dtype=weight.dtype,
                                    prune=prune)
    n, _k = int(weight.shape[0]), int(weight.shape[1])
    result = {
        f"{name}.sparse_weight": q.sparse_weight.cpu().contiguous(),
        f"{name}.metadata": q.metadata.cpu().contiguous(),
        f"{name}.weight_scale": q.scale.cpu().contiguous(),
        f"{name}.weight_global_scale": q.global_scale.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    if n <= 0:
        raise ValueError(f"{name} has empty output dimension")
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


def quantize_weight(method: str, name: str, weight: torch.Tensor,
                    prune: bool) -> dict[str, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be a 2D linear weight")
    if method == "sparse_nvfp4_mytest":
        return quantize_sparse_nvfp4(name, weight, prune)
    if method == "sparse_bf16_mytest":
        return quantize_sparse_bf16(name, weight, prune)
    raise ValueError(f"unknown method: {method}")


def layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def export(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sparse export")
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
    update_config(output_dir, args.method)

    with (model_path / "config.json").open() as f:
        config = json.load(f)
    num_layers = int(config["num_hidden_layers"])

    weight_map = read_weight_map(model_path)
    reader = LazyTensorReader(model_path, weight_map)
    linear_sources: set[str] = set()
    output_tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []

    def convert(base_name: str, source: str, weight: torch.Tensor) -> None:
        try:
            output_tensors.update(
                quantize_weight(args.method, base_name, weight, args.prune))
            manifest[base_name] = {
                "source": source,
                "shape": list(weight.shape),
                "prune": bool(args.prune),
            }
        except Exception as exc:
            failures.append({
                "name": base_name,
                "source": source,
                "error": repr(exc),
            })
            with (output_dir / f"{args.method}_failures.json").open("w") as f:
                json.dump(failures, f, indent=2, sort_keys=True)
                f.write("\n")
            raise

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
                fused_qkv)
        del fused_qkv

        fused_gate_up = torch.cat([reader.get(name) for name in gate_up_names],
                                  dim=0)
        convert(f"{prefix}.mlp.gate_up_proj", "gate_proj+up_proj",
                fused_gate_up)
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
    with (output_dir / f"{args.method}_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    if failures:
        with (output_dir / f"{args.method}_failures.json").open("w") as f:
            json.dump(failures, f, indent=2, sort_keys=True)
            f.write("\n")

    total_bytes = os.path.getsize(save_path)
    print(f"saved {save_path} ({total_bytes / 1024**3:.2f} GiB)")


def main(method: str) -> None:
    parser = make_parser(method)
    args = parser.parse_args()
    try:
        export(args)
    except Exception:
        failure_path = Path(args.output_dir).resolve() / f"{method}_failures.json"
        if not failure_path.exists() and Path(args.output_dir).exists():
            with failure_path.open("w") as f:
                json.dump([{"error": "export failed before layer manifest"}], f)
                f.write("\n")
        raise
