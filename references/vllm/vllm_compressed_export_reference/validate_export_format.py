#!/usr/bin/env python3
"""Validate exported checkpoint naming for this repo's vLLM compression paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


CUSTOM_SUFFIXES = {
    "nvfp4_mytest": ("weight", "weight_scale", "weight_global_scale"),
    "sparse_bf16_mytest": ("sparse_weight", "metadata"),
    "sparse_nvfp4_mytest": (
        "sparse_weight",
        "metadata",
        "weight_scale",
        "weight_global_scale",
    ),
    "marlin_nvfp4_mytest": (
        "packed_weight",
        "weight_scale",
        "weight_global_scale",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-errors", type=int, default=40)
    return parser.parse_args()


def load_config(model_dir: Path) -> dict:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def safetensor_keys(model_dir: Path) -> set[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted({model_dir / name for name in weight_map.values()})
    else:
        files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    keys: set[str] = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys.update(f.keys())
    return keys


def fused_bases(layer: int) -> list[str]:
    prefix = f"model.layers.{layer}"
    return [
        f"{prefix}.self_attn.qkv_proj",
        f"{prefix}.self_attn.o_proj",
        f"{prefix}.mlp.gate_up_proj",
        f"{prefix}.mlp.down_proj",
    ]


def nofuse_bases(layer: int) -> list[str]:
    prefix = f"model.layers.{layer}"
    return [
        f"{prefix}.self_attn.q_proj",
        f"{prefix}.self_attn.k_proj",
        f"{prefix}.self_attn.v_proj",
        f"{prefix}.self_attn.o_proj",
        f"{prefix}.mlp.gate_proj",
        f"{prefix}.mlp.up_proj",
        f"{prefix}.mlp.down_proj",
    ]


def compressed_tensors_required(layer: int, nofuse: bool) -> list[str]:
    required: list[str] = []
    if nofuse:
        for base in nofuse_bases(layer):
            required.extend([
                f"{base}.weight_packed",
                f"{base}.weight_scale",
                f"{base}.weight_global_scale",
            ])
        return required

    prefix = f"model.layers.{layer}"
    required.extend([
        f"{prefix}.self_attn.qkv_proj.weight_packed",
        f"{prefix}.self_attn.qkv_proj.weight_scale",
        f"{prefix}.self_attn.q_proj.weight_global_scale",
        f"{prefix}.self_attn.k_proj.weight_global_scale",
        f"{prefix}.self_attn.v_proj.weight_global_scale",
        f"{prefix}.self_attn.o_proj.weight_packed",
        f"{prefix}.self_attn.o_proj.weight_scale",
        f"{prefix}.self_attn.o_proj.weight_global_scale",
        f"{prefix}.mlp.gate_up_proj.weight_packed",
        f"{prefix}.mlp.gate_up_proj.weight_scale",
        f"{prefix}.mlp.gate_proj.weight_global_scale",
        f"{prefix}.mlp.up_proj.weight_global_scale",
        f"{prefix}.mlp.down_proj.weight_packed",
        f"{prefix}.mlp.down_proj.weight_scale",
        f"{prefix}.mlp.down_proj.weight_global_scale",
    ])
    return required


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    config = load_config(model_dir)
    keys = safetensor_keys(model_dir)

    qconfig = config.get("quantization_config") or {}
    method = qconfig.get("quant_method")
    if not method:
        raise SystemExit("config.json has no quantization_config.quant_method")

    num_layers = int(config["num_hidden_layers"])
    nofuse = "LlamaNoFuseForCausalLM" in config.get("architectures", [])
    bases_fn = nofuse_bases if nofuse else fused_bases

    required: list[str] = [
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    if method == "compressed-tensors":
        for layer in range(num_layers):
            required.extend(compressed_tensors_required(layer, nofuse))
    elif method in CUSTOM_SUFFIXES:
        suffixes = CUSTOM_SUFFIXES[method]
        for layer in range(num_layers):
            for base in bases_fn(layer):
                required.extend(f"{base}.{suffix}" for suffix in suffixes)
    else:
        raise SystemExit(f"Unsupported quant_method for this validator: {method}")

    if "lm_head" not in qconfig.get("modules_to_not_convert", []) and (
            "lm_head" not in qconfig.get("ignore", [])):
        print("warning: lm_head is not listed as skipped/ignored in config")
    if "lm_head.weight" not in keys:
        print("warning: lm_head.weight not found; this may be OK for tied embeddings")

    missing = [name for name in required if name not in keys]
    print(f"model_dir={model_dir}")
    print(f"quant_method={method}")
    print(f"architecture={'nofuse' if nofuse else 'fused'}")
    print(f"num_layers={num_layers}")
    print(f"safetensors_keys={len(keys)}")

    if missing:
        print(f"missing_required_tensors={len(missing)}")
        for name in missing[:args.max_errors]:
            print(f"missing: {name}")
        if len(missing) > args.max_errors:
            print(f"... {len(missing) - args.max_errors} more missing tensors")
        return 1

    print("format_check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
