#!/usr/bin/env python3
"""Export a selected 068 decode checkpoint for phase_hetero_mytest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_CUTLASS_WRAPPER = str(
    Path(os.environ.get("COSPAQ_REPO_ROOT", Path(__file__).resolve().parents[4]))
    / "fake/kernels/cutlass/cutlass_wrapper")

SUPPORTED_METHODS = {
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "w4a16_ours",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-json")
    parser.add_argument("--default-prefill-method", default="dense_nvfp4")
    parser.add_argument("--default-decode-method", default="w4a16_ours")
    parser.add_argument("--cutlass-wrapper-path", default=DEFAULT_CUTLASS_WRAPPER)
    parser.add_argument("--canonical-sparse-bf16-state", type=Path)
    parser.add_argument("--canonical-sparse-nvfp4-state", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prune", action="store_true",
                        help="Forbidden in 068; retained only for fail-fast compatibility.")
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


class CanonicalTensorReader:

    def __init__(self, path: Path, method: str) -> None:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        metadata = payload.get("metadata", {})
        if metadata.get("method") != method:
            raise RuntimeError(
                f"{path}: expected method={method}, got {metadata.get('method')}")
        if method == "sparse_nvfp4" and metadata.get(
                "sparse_nvfp4_prequant_only") is not True:
            raise RuntimeError(f"{path}: sparse NVFP4 state is not prequant-only")
        self.path = path.resolve()
        self.metadata = metadata
        self.state = payload["state_dict"]

    def get(self, name: str) -> torch.Tensor:
        return self.state[name]


def layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def build_demo_policy(num_layers: int, default_prefill: str,
                      default_decode: str) -> dict[str, Any]:
    methods = [
        "dense_bf16",
        "sparse_bf16",
        "dense_nvfp4",
        "sparse_nvfp4",
        "w4a16_ours",
    ]
    method_map: dict[str, dict[str, str]] = {}
    suffixes = [
        "self_attn.qkv_proj",
        "self_attn.o_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
    ]
    cursor = 0
    for layer_idx in range(num_layers):
        prefix = layer_prefix(layer_idx)
        for suffix in suffixes:
            method_map[f"{prefix}.{suffix}"] = {
                "prefill_method": methods[cursor % len(methods)],
                "decode_method": methods[(cursor + 2) % len(methods)],
            }
            cursor += 1
    return {
        "default_prefill_method": default_prefill,
        "default_decode_method": default_decode,
        "modules_to_not_convert": ["lm_head"],
        "method_map": method_map,
    }


def normalize_policy(raw: dict[str, Any], default_prefill: str,
                     default_decode: str) -> dict[str, Any]:
    policy = dict(raw)
    policy.setdefault("default_prefill_method",
                      policy.pop("prefill_method", default_prefill))
    policy.setdefault("default_decode_method",
                      policy.pop("decode_method", default_decode))
    policy.setdefault("modules_to_not_convert", ["lm_head"])
    policy.setdefault("method_map", {})
    for field in ("default_prefill_method", "default_decode_method"):
        method = str(policy[field])
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported {field}: {method}")
        policy[field] = method
    if not isinstance(policy["method_map"], dict):
        raise ValueError("policy method_map must be a dict")
    normalized: dict[str, dict[str, str]] = {}
    for prefix, value in policy["method_map"].items():
        if isinstance(value, str):
            prefill_method = value
            decode_method = value
        elif isinstance(value, dict):
            prefill_method = str(
                value.get("prefill_method", policy["default_prefill_method"]))
            decode_method = str(
                value.get("decode_method", policy["default_decode_method"]))
        else:
            raise ValueError(f"method_map[{prefix!r}] must be string or dict")
        if prefill_method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported prefill method for {prefix}: "
                             f"{prefill_method}")
        if decode_method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported decode method for {prefix}: "
                             f"{decode_method}")
        normalized[str(prefix)] = {
            "prefill_method": prefill_method,
            "decode_method": decode_method,
        }
    policy["method_map"] = normalized
    return policy


def load_policy(path: str | None, num_layers: int, default_prefill: str,
                default_decode: str) -> dict[str, Any]:
    if path is None:
        raw = {
            "default_prefill_method": default_prefill,
            "default_decode_method": default_decode,
            "method_map": {},
            "modules_to_not_convert": ["lm_head"],
        }
    elif path == "demo":
        raw = build_demo_policy(num_layers, default_prefill, default_decode)
    else:
        with Path(path).open() as f:
            raw = json.load(f)
    return normalize_policy(raw, default_prefill, default_decode)


def update_config(output_dir: Path, policy: dict[str, Any]) -> None:
    config_path = output_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    config["quantization_config"] = {
        "quant_method": "phase_hetero_mytest",
        "default_prefill_method": policy["default_prefill_method"],
        "default_decode_method": policy["default_decode_method"],
        "modules_to_not_convert": policy["modules_to_not_convert"],
        "method_map": policy["method_map"],
    }
    config["torch_dtype"] = "bfloat16"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


@torch.no_grad()
def convert_dense_bf16(name: str,
                       weight: torch.Tensor) -> dict[str, torch.Tensor]:
    return {f"{name}.weight": weight.to(torch.bfloat16).cpu().contiguous()}


@torch.no_grad()
def quantize_dense_nvfp4(name: str,
                         weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import quantize_weight_bf16

    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n % 32 != 0 or k % 32 != 0:
        raise ValueError(f"{name} dense_nvfp4 requires N,K divisible by 32")
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
def quantize_w4a16_ours(name: str,
                        weight: torch.Tensor) -> dict[str, torch.Tensor]:
    from cutlass_wrapper import pack_marlin_nvfp4_weight

    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n % 64 != 0 or k % 128 != 0:
        raise ValueError(f"{name} w4a16_ours requires N%64==0 and K%128==0")
    cuda_weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    q = pack_marlin_nvfp4_weight(cuda_weight,
                                 activation_dtype=torch.bfloat16,
                                 original_dtype=weight.dtype)
    result = {
        f"{name}.packed_weight": q.packed_weight.cpu().contiguous(),
        f"{name}.weight_scale": q.weight_scale.cpu().contiguous(),
        f"{name}.weight_global_scale": q.global_scale.cpu().contiguous(),
    }
    del cuda_weight, q
    torch.cuda.empty_cache()
    return result


def convert_method(method: str, name: str, weight: torch.Tensor,
                   prune: bool) -> dict[str, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be a 2D linear weight")
    if method == "dense_bf16":
        return convert_dense_bf16(name, weight)
    if method == "dense_nvfp4":
        return quantize_dense_nvfp4(name, weight)
    if method == "sparse_bf16":
        return quantize_sparse_bf16(name, weight, prune)
    if method == "sparse_nvfp4":
        return quantize_sparse_nvfp4(name, weight, prune)
    if method == "w4a16_ours":
        return quantize_w4a16_ours(name, weight)
    raise AssertionError(f"unsupported method: {method}")


def phase_prefix_tensors(tensors: dict[str, torch.Tensor], base_name: str,
                         phase_prefix: str) -> dict[str, torch.Tensor]:
    old_prefix = f"{base_name}."
    new_prefix = f"{base_name}.{phase_prefix}_"
    return {
        f"{new_prefix}{key.removeprefix(old_prefix)}": value
        for key, value in tensors.items()
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.prune:
        raise RuntimeError("068 forbids direct pruning; provide canonical sparse states")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for phase-hetero export")
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

    with (model_path / "config.json").open() as f:
        config = json.load(f)
    num_layers = int(config["num_hidden_layers"])
    policy = load_policy(args.policy_json, num_layers,
                         args.default_prefill_method,
                         args.default_decode_method)
    update_config(output_dir, policy)

    weight_map = read_weight_map(model_path)
    reader = LazyTensorReader(model_path, weight_map)
    method_map: dict[str, dict[str, str]] = policy["method_map"]
    default_prefill = str(policy["default_prefill_method"])
    default_decode = str(policy["default_decode_method"])
    linear_sources: set[str] = set()
    output_tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, dict[str, Any]] = {}

    used_methods = {default_prefill, default_decode}
    for entry in method_map.values():
        used_methods.update((entry["prefill_method"], entry["decode_method"]))
    canonical_readers: dict[str, CanonicalTensorReader] = {}
    for method, path in (("sparse_bf16", args.canonical_sparse_bf16_state),
                         ("sparse_nvfp4", args.canonical_sparse_nvfp4_state)):
        if method in used_methods:
            if path is None or not path.is_file():
                raise FileNotFoundError(
                    f"policy uses {method}, but its canonical state is missing")
            canonical_readers[method] = CanonicalTensorReader(path, method)

    def methods_for(base_name: str) -> tuple[str, str]:
        item = method_map.get(base_name)
        if item is None:
            return default_prefill, default_decode
        return item["prefill_method"], item["decode_method"]

    def weight_for(method: str, source_names: list[str]) -> torch.Tensor:
        source_reader = canonical_readers.get(method, reader)
        tensors = [source_reader.get(name) for name in source_names]
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)

    def convert(base_name: str, source: str, source_names: list[str]) -> None:
        prefill_method, decode_method = methods_for(base_name)
        prefill_weight = weight_for(prefill_method, source_names)
        manifest[base_name] = {
            "source": source,
            "shape": list(prefill_weight.shape),
            "prefill_method": prefill_method,
            "decode_method": decode_method,
            "same_method": prefill_method == decode_method,
            "prefill_weight_source": "canonical" if prefill_method.startswith("sparse_") else "original_dense",
            "decode_weight_source": "canonical" if decode_method.startswith("sparse_") else "original_dense",
        }
        prefill = convert_method(prefill_method, base_name, prefill_weight, False)
        output_tensors.update(
            phase_prefix_tensors(prefill, base_name, "ph_prefill"))
        if prefill_method != decode_method:
            decode_weight = weight_for(decode_method, source_names)
            decode = convert_method(decode_method, base_name, decode_weight, False)
            output_tensors.update(
                phase_prefix_tensors(decode, base_name, "ph_decode"))
            del decode_weight
        del prefill_weight

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

        convert(f"{prefix}.self_attn.qkv_proj", "q_proj+k_proj+v_proj",
                qkv_names)
        convert(f"{prefix}.mlp.gate_up_proj", "gate_proj+up_proj",
                gate_up_names)

        for suffix in ("self_attn.o_proj", "mlp.down_proj"):
            source_name = f"{prefix}.{suffix}.weight"
            linear_sources.add(source_name)
            convert(f"{prefix}.{suffix}", source_name, [source_name])

        print(f"converted layer {layer_idx + 1}/{num_layers}", flush=True)

    for name in sorted(weight_map):
        if name in linear_sources:
            continue
        output_tensors[name] = reader.get(name).contiguous()

    save_path = output_dir / "model.safetensors"
    save_file(output_tensors, save_path, metadata={"format": "pt"})
    with (output_dir / "phase_hetero_policy.json").open("w") as f:
        json.dump(policy, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "phase_hetero_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    provenance = {
        "direct_prune": False,
        "model_path": str(model_path),
        "policy_json": str(Path(args.policy_json).resolve()) if args.policy_json not in (None, "demo") else args.policy_json,
        "canonical_sparse_bf16_state": str(canonical_readers["sparse_bf16"].path) if "sparse_bf16" in canonical_readers else None,
        "canonical_sparse_nvfp4_state": str(canonical_readers["sparse_nvfp4"].path) if "sparse_nvfp4" in canonical_readers else None,
        "sparse_nvfp4_prequant_only": canonical_readers["sparse_nvfp4"].metadata.get("sparse_nvfp4_prequant_only") if "sparse_nvfp4" in canonical_readers else None,
    }
    with (output_dir / "phase_hetero_provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    counts: dict[str, int] = {}
    for item in manifest.values():
        key = f"{item['prefill_method']}->{item['decode_method']}"
        counts[key] = counts.get(key, 0) + 1
    total_bytes = os.path.getsize(save_path)
    print(f"phase_counts={json.dumps(counts, sort_keys=True)}")
    print(f"saved {save_path} ({total_bytes / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()
