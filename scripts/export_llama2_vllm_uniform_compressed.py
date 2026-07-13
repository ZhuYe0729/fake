#!/usr/bin/env python3
"""Export Llama2-7B uniform compressed checkpoints for local vLLM backends."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_ROOT = (
    REPO_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/exports/vllm/llama2_7b_018"
DEFAULT_CUTLASS_WRAPPER = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
METHOD_ORDER = ("sparse_bf16", "dense_nvfp4", "sparse_nvfp4")


@dataclass(frozen=True)
class MethodSpec:
    method: str
    output_name: str
    quant_method: str
    manifest_name: str


METHOD_SPECS = {
    "dense_nvfp4": MethodSpec(
        method="dense_nvfp4",
        output_name="uniform_dense_nvfp4",
        quant_method="nvfp4_mytest",
        manifest_name="nvfp4_mytest_manifest.json",
    ),
    "sparse_bf16": MethodSpec(
        method="sparse_bf16",
        output_name="uniform_sparse_bf16",
        quant_method="sparse_bf16_mytest",
        manifest_name="sparse_bf16_mytest_manifest.json",
    ),
    "sparse_nvfp4": MethodSpec(
        method="sparse_nvfp4",
        output_name="uniform_sparse_nvfp4",
        quant_method="sparse_nvfp4_mytest",
        manifest_name="sparse_nvfp4_mytest_manifest.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export prepared Llama2 compressed artifacts to vLLM format."
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cutlass-wrapper-path", type=Path, default=DEFAULT_CUTLASS_WRAPPER)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned exports without loading weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()
    model_path = resolve_model_path(args.model_path, prepared_root, methods).resolve()

    validate_inputs(model_path, prepared_root, methods)
    if args.dry_run:
        print_plan(model_path, prepared_root, output_root, methods)
        return

    require_cuda()
    sys.path.insert(0, str(args.cutlass_wrapper_path.resolve()))

    with (model_path / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
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


def resolve_model_path(
    model_path: Path | None, prepared_root: Path, methods: list[str]
) -> Path:
    if model_path is not None:
        return model_path
    for method in methods:
        metadata_path = prepared_root / method / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        path = metadata.get("model_path")
        if path:
            return Path(path)
    raise FileNotFoundError(
        "could not infer --model-path from prepared metadata; pass --model-path explicitly"
    )


def validate_inputs(model_path: Path, prepared_root: Path, methods: list[str]) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"missing config.json under model path: {model_path}")
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


def print_plan(
    model_path: Path, prepared_root: Path, output_root: Path, methods: list[str]
) -> None:
    print(f"model_path={model_path}")
    print(f"prepared_root={prepared_root}")
    print(f"output_root={output_root}")
    for method in methods:
        spec = METHOD_SPECS[method]
        print(
            f"export {method}: prepared={prepared_root / method / 'model.pt'} "
            f"output={output_root / spec.output_name} quant_method={spec.quant_method}"
        )


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for compressed vLLM export")
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        raise RuntimeError(
            f"SM120-class GPU is required for this export path, got {major}.{minor}"
        )


def read_weight_map(model_path: Path) -> dict[str, str]:
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        safetensors = sorted(model_path.glob("*.safetensors"))
        if len(safetensors) != 1:
            raise FileNotFoundError(
                "expected model.safetensors.index.json or exactly one safetensors file"
            )
        with safe_open(safetensors[0], framework="pt", device="cpu") as f:
            return {key: safetensors[0].name for key in f.keys()}
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


class LazyTensorReader:
    def __init__(self, model_path: Path, weight_map: dict[str, str]) -> None:
        self.model_path = model_path
        self.weight_map = weight_map

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        filename = self.weight_map[name]
        with safe_open(self.model_path / filename, framework="pt", device="cpu") as f:
            return f.get_tensor(name)


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


def update_config(output_dir: Path, spec: MethodSpec) -> None:
    config_path = output_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
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
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def load_prepared_state(prepared_root: Path, method: str) -> dict[str, torch.Tensor]:
    path = prepared_root / method / "model.pt"
    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("method") != method:
        raise RuntimeError(
            f"prepared artifact method mismatch: expected={method} got={metadata.get('method')}"
        )
    return payload["state_dict"]


def layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


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
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} already exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    copy_model_assets(model_path, output_dir)
    update_config(output_dir, spec)

    print(f"loading prepared state for {spec.method}", flush=True)
    prepared_state = load_prepared_state(prepared_root, spec.method)
    output_tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, dict[str, Any]] = {}
    linear_sources: set[str] = set()

    def prepared(name: str) -> torch.Tensor:
        if name not in prepared_state:
            raise KeyError(f"missing prepared tensor: {name}")
        return prepared_state[name]

    def convert(base_name: str, source: str, weight: torch.Tensor) -> None:
        output_tensors.update(quantize_weight(spec.method, base_name, weight))
        manifest[base_name] = {
            "method": spec.method,
            "quant_method": spec.quant_method,
            "source": source,
            "shape": list(weight.shape),
            "prune": False,
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
        linear_sources.update(qkv_names)
        linear_sources.update(gate_up_names)

        fused_qkv = torch.cat([prepared(name) for name in qkv_names], dim=0)
        convert(f"{prefix}.self_attn.qkv_proj", "q_proj+k_proj+v_proj", fused_qkv)
        del fused_qkv

        fused_gate_up = torch.cat([prepared(name) for name in gate_up_names], dim=0)
        convert(f"{prefix}.mlp.gate_up_proj", "gate_proj+up_proj", fused_gate_up)
        del fused_gate_up

        for suffix in ("self_attn.o_proj", "mlp.down_proj"):
            source_name = f"{prefix}.{suffix}.weight"
            linear_sources.add(source_name)
            convert(f"{prefix}.{suffix}", source_name, prepared(source_name))

        print(f"converted {spec.output_name} layer {i + 1}/{num_layers}", flush=True)

    for name in sorted(weight_map):
        if name in linear_sources:
            continue
        output_tensors[name] = reader.get(name).contiguous()

    save_path = output_dir / "model.safetensors"
    from safetensors.torch import save_file

    save_file(output_tensors, save_path, metadata={"format": "pt"})
    with (output_dir / spec.manifest_name).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    write_export_summary(output_dir, spec, save_path, manifest)
    print(f"saved {save_path} ({os.path.getsize(save_path) / 1024**3:.2f} GiB)")


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
    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def quantize_dense_nvfp4(name: str, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import quantize_weight_bf16

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
    q = quantize_sparse_weight_bf16(
        cuda_weight, original_dtype=weight.dtype, prune=False
    )
    result = {
        f"{name}.sparse_weight": q.sparse_weight.cpu().contiguous(),
        f"{name}.metadata": q.metadata.cpu().contiguous(),
        f"{name}.weight_scale": q.scale.cpu().contiguous(),
        f"{name}.weight_global_scale": q.global_scale.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


def write_export_summary(
    output_dir: Path,
    spec: MethodSpec,
    save_path: Path,
    manifest: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "checkpoint_format": "llama2_vllm_uniform_compressed_fused_v1",
        "method": spec.method,
        "quant_method": spec.quant_method,
        "architecture": "fused_llama",
        "linear_bases": len(manifest),
        "model_safetensors": str(save_path),
        "model_safetensors_gib": os.path.getsize(save_path) / 1024**3,
    }
    with (output_dir / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
