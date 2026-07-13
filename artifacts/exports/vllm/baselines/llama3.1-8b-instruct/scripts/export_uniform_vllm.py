#!/usr/bin/env python3
"""Export Llama-3.1-8B-Instruct uniform compressed artifacts to fused vLLM checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BASELINE_ROOT.parents[4]
MODEL_PATH = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
DEFAULT_CUTLASS_WRAPPER = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
METHOD_ORDER = ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")


@dataclass(frozen=True)
class MethodSpec:
    method: str
    output_name: str
    quant_method: str
    manifest_name: str


METHOD_SPECS = {
    "dense_nvfp4": MethodSpec(
        "dense_nvfp4",
        "uniform_dense_nvfp4",
        "nvfp4_mytest",
        "nvfp4_mytest_manifest.json",
    ),
    "sparse_bf16": MethodSpec(
        "sparse_bf16",
        "uniform_sparse_bf16",
        "sparse_bf16_mytest",
        "sparse_bf16_mytest_manifest.json",
    ),
    "sparse_nvfp4": MethodSpec(
        "sparse_nvfp4",
        "uniform_sparse_nvfp4",
        "sparse_nvfp4_mytest",
        "sparse_nvfp4_mytest_manifest.json",
    ),
    "marlin_nvfp4": MethodSpec(
        "marlin_nvfp4",
        "uniform_marlin_nvfp4",
        "marlin_nvfp4_mytest",
        "marlin_nvfp4_mytest_manifest.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--prepared-root", type=Path, default=BASELINE_ROOT / "prepared")
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT / "checkpoints")
    parser.add_argument("--cutlass-wrapper-path", type=Path, default=DEFAULT_CUTLASS_WRAPPER)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    model_path = args.model_path.resolve()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()
    validate_inputs(model_path, prepared_root, methods)

    output_root.mkdir(parents=True, exist_ok=True)
    write_dense_manifest(model_path, output_root)
    if args.dry_run:
        print_plan(model_path, prepared_root, output_root, methods)
        return

    require_cuda()
    sys.path.insert(0, str(args.cutlass_wrapper_path.resolve()))

    config = json.loads((model_path / "config.json").read_text())
    num_layers = int(config["num_hidden_layers"])
    weight_map = read_weight_map(model_path)
    reader = LazyTensorReader(model_path, weight_map)

    for method in methods:
        spec = METHOD_SPECS[method]
        export_one_method(
            spec=spec,
            model_path=model_path,
            prepared_root=prepared_root,
            output_root=output_root,
            reader=reader,
            weight_map=weight_map,
            num_layers=num_layers,
            force=args.force,
        )


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={sorted(METHOD_SPECS)}")
    return methods


def validate_inputs(model_path: Path, prepared_root: Path, methods: list[str]) -> None:
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"missing model config: {model_path / 'config.json'}")
    for method in methods:
        for name in ("model.pt", "metadata.json"):
            path = prepared_root / method / name
            if not path.exists():
                raise FileNotFoundError(path)
        metadata = json.loads((prepared_root / method / "metadata.json").read_text())
        if metadata.get("method") != method:
            raise RuntimeError(
                f"prepared metadata method mismatch for {method}: {metadata.get('method')}"
            )


def print_plan(model_path: Path, prepared_root: Path, output_root: Path, methods: list[str]) -> None:
    print(f"model_path={model_path}")
    print(f"prepared_root={prepared_root}")
    print(f"output_root={output_root}")
    print(f"dense_bf16 -> {model_path}")
    for method in methods:
        spec = METHOD_SPECS[method]
        print(f"{method}: {prepared_root / method / 'model.pt'} -> {output_root / spec.output_name}")


def write_dense_manifest(model_path: Path, output_root: Path) -> None:
    payload = {
        "method": "dense_bf16",
        "model_path": str(model_path),
        "checkpoint_format": "hf_dense_bf16_reference",
    }
    (output_root / "dense_bf16_model.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for compressed vLLM export")
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        raise RuntimeError(f"SM120-class GPU is required for this export path, got {major}.{minor}")


def read_weight_map(model_path: Path) -> dict[str, str]:
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        safetensors = sorted(model_path.glob("*.safetensors"))
        if len(safetensors) != 1:
            raise FileNotFoundError("expected one safetensors file or an index")
        with safe_open(safetensors[0], framework="pt", device="cpu") as handle:
            return {key: safetensors[0].name for key in handle.keys()}
    return json.loads(index_path.read_text())["weight_map"]


class LazyTensorReader:
    def __init__(self, model_path: Path, weight_map: dict[str, str]) -> None:
        self.model_path = model_path
        self.weight_map = weight_map

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(self.model_path / self.weight_map[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)


def copy_model_assets(src: Path, dst: Path) -> None:
    skip_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    skip_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    for item in src.iterdir():
        if item.name in skip_names:
            continue
        target = dst / item.name
        if item.is_file() and item.suffix in skip_suffixes:
            continue
        if item.is_dir():
            if not item.name.startswith("."):
                shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)


def update_config(output_dir: Path, spec: MethodSpec) -> None:
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    qconfig: dict[str, Any] = {
        "quant_method": spec.quant_method,
        "modules_to_not_convert": ["lm_head"],
    }
    if spec.method == "dense_nvfp4":
        qconfig["group_size"] = 16
    if spec.method == "sparse_bf16":
        qconfig["backend"] = "cusparselt"
    config["quantization_config"] = qconfig
    config["torch_dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def load_prepared_state(prepared_root: Path, method: str) -> dict[str, torch.Tensor]:
    payload = torch.load(prepared_root / method / "model.pt", map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("method") != method:
        raise RuntimeError(f"prepared artifact method mismatch: expected={method} got={metadata.get('method')}")
    return payload["state_dict"]


def reset_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def source_linear_names(layer_idx: int) -> dict[str, list[str]]:
    prefix = layer_prefix(layer_idx)
    return {
        "self_attn.qkv_proj": [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ],
        "self_attn.o_proj": [f"{prefix}.self_attn.o_proj.weight"],
        "mlp.gate_up_proj": [
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
        ],
        "mlp.down_proj": [f"{prefix}.mlp.down_proj.weight"],
    }


def fused_weight_from_state(state: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    tensors = [state[name] for name in names]
    if len(tensors) == 1:
        return tensors[0].contiguous()
    return torch.cat(tensors, dim=0).contiguous()


def export_one_method(
    *,
    spec: MethodSpec,
    model_path: Path,
    prepared_root: Path,
    output_root: Path,
    reader: LazyTensorReader,
    weight_map: dict[str, str],
    num_layers: int,
    force: bool,
) -> None:
    output_dir = output_root / spec.output_name
    reset_output_dir(output_dir, force)
    copy_model_assets(model_path, output_dir)
    update_config(output_dir, spec)
    prepared = load_prepared_state(prepared_root, spec.method)
    output_tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, dict[str, Any]] = {}
    linear_sources: set[str] = set()

    for i in range(num_layers):
        for fused_suffix, names in source_linear_names(i).items():
            base_name = f"{layer_prefix(i)}.{fused_suffix}"
            linear_sources.update(names)
            weight = fused_weight_from_state(prepared, names)
            output_tensors.update(quantize_weight(spec.method, base_name, weight))
            manifest[base_name] = {
                "method": spec.method,
                "quant_method": spec.quant_method,
                "source": "+".join(names),
                "shape": list(weight.shape),
                "prune": False,
            }
            del weight
        print(f"converted {spec.output_name} layer {i + 1}/{num_layers}", flush=True)

    for name in sorted(weight_map):
        if name in linear_sources:
            continue
        output_tensors[name] = reader.get(name).contiguous()

    save_export(output_dir, output_tensors, spec.manifest_name, manifest)
    write_export_summary(output_dir, spec, len(manifest))


@torch.no_grad()
def quantize_weight(method: str, name: str, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be a 2D linear weight")
    if method == "dense_nvfp4":
        return quantize_dense_nvfp4(name, weight)
    if method == "sparse_bf16":
        return quantize_sparse_bf16(name, weight)
    if method == "sparse_nvfp4":
        return quantize_sparse_nvfp4(name, weight)
    if method == "marlin_nvfp4":
        return quantize_marlin_nvfp4(name, weight)
    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def quantize_dense_nvfp4(name: str, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import quantize_weight_bf16

    n, k = int(weight.shape[0]), int(weight.shape[1])
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
def quantize_sparse_bf16(name: str, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import pack_sparse_bf16_weight

    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = pack_sparse_bf16_weight(
        cuda_weight, original_dtype=weight.dtype, prune=False, backend="cusparselt"
    )
    result = {
        f"{name}.sparse_weight": q.sparse_weight.cpu().contiguous(),
        f"{name}.metadata": q.metadata.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def quantize_sparse_nvfp4(name: str, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import quantize_sparse_weight_bf16

    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = quantize_sparse_weight_bf16(cuda_weight, original_dtype=weight.dtype, prune=False)
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
def quantize_marlin_nvfp4(name: str, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import pack_marlin_nvfp4_weight

    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = pack_marlin_nvfp4_weight(
        cuda_weight, activation_dtype=torch.bfloat16, original_dtype=weight.dtype
    )
    result = {
        f"{name}.packed_weight": q.packed_weight.cpu().contiguous(),
        f"{name}.weight_scale": q.weight_scale.cpu().contiguous(),
        f"{name}.weight_global_scale": q.global_scale.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


def save_export(
    output_dir: Path,
    output_tensors: dict[str, torch.Tensor],
    manifest_name: str,
    manifest: dict[str, dict[str, Any]],
) -> None:
    from safetensors.torch import save_file

    save_path = output_dir / "model.safetensors"
    save_file(output_tensors, save_path, metadata={"format": "pt"})
    with (output_dir / manifest_name).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"saved {save_path} ({os.path.getsize(save_path) / 1024**3:.2f} GiB)", flush=True)


def write_export_summary(output_dir: Path, spec: MethodSpec, linear_bases: int) -> None:
    payload = {
        "checkpoint_format": "llama31_instruct_vllm_uniform_compressed_fused_v1",
        "method": spec.method,
        "quant_method": spec.quant_method,
        "architecture": "fused_llama",
        "linear_bases": linear_bases,
        "model_safetensors": str(output_dir / "model.safetensors"),
        "model_safetensors_gib": os.path.getsize(output_dir / "model.safetensors") / 1024**3,
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
