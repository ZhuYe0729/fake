#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = FAKE_ROOT.parent
SOURCE_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy"
QUALITY_007_SCRIPTS = DEBUG_ROOT.parent / "007_llama2_quality_modeling" / "scripts"

for path in (WORKSPACE_ROOT, FAKE_ROOT, QUALITY_007_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_quality import (  # type: ignore  # noqa: E402
    MODEL_KEY,
    QualityConfig,
    cleanup_cuda,
    compressible_modules,
    compute_nll,
    dtype_from_arg,
    layer_index,
    load_calibration_blocks,
    load_llama_for_quality,
    local_cuda_index,
    module_family,
    module_record,
    module_type,
    parse_policy,
    weight_stats,
)
from common import module_parent, utc_now  # type: ignore  # noqa: E402


METHODS = ("dense_nvfp4", "sparse_nvfp4")
LINEAR_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LAYER_COUNT = 32


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={METHODS}")
    return methods


def quality_config_from_args(args: Any) -> QualityConfig:
    return QualityConfig(
        calib_samples=args.calib_samples,
        seq_len=args.seq_len,
        seed=args.seed,
        batch_size=args.batch_size,
        source_root=args.source_root,
        output_root=args.output_root,
    )


def select_modules(modules: list[Any], max_modules: int | None) -> list[Any]:
    if max_modules is None:
        return modules
    return modules[: max_modules]


def default_loss_policies(module_names: Iterable[str]) -> list[dict[str, Any]]:
    names = list(module_names)
    policies: list[dict[str, Any]] = []
    for layer in range(LAYER_COUNT):
        selected = {name for name in names if layer_index(name) == layer}
        if selected:
            policies.append({"policy": f"layer:{layer}", "policy_kind": "layer", "layer": layer, "linear_type": "", "selected": selected})
    for typ in LINEAR_TYPES:
        selected = {name for name in names if module_type(name) == typ}
        if selected:
            policies.append({"policy": f"type:{typ}", "policy_kind": "type", "layer": "", "linear_type": typ, "selected": selected})
    for layer in range(LAYER_COUNT):
        for typ in LINEAR_TYPES:
            selected = {name for name in names if layer_index(name) == layer and module_type(name) == typ}
            if selected:
                policies.append(
                    {
                        "policy": f"layer_type:{layer}:{typ}",
                        "policy_kind": "layer_type",
                        "layer": layer,
                        "linear_type": typ,
                        "selected": selected,
                    }
                )
    return policies


def parse_loss_policies(spec: str, module_names: Iterable[str]) -> list[dict[str, Any]]:
    if spec == "default":
        return default_loss_policies(module_names)
    out = []
    names = list(module_names)
    for item in [part.strip() for part in spec.split(",") if part.strip()]:
        if item.startswith("layer_type:"):
            _, layer_text, typ = item.split(":", 2)
            layer = int(layer_text)
            selected = {name for name in names if layer_index(name) == layer and module_type(name) == typ}
            out.append({"policy": item, "policy_kind": "layer_type", "layer": layer, "linear_type": typ, "selected": selected})
        else:
            selected = parse_policy(item, names)
            kind = item.split(":", 1)[0] if ":" in item else item
            out.append({"policy": item, "policy_kind": kind, "layer": "", "linear_type": "", "selected": selected})
    return out


def load_prepared_payload(source_root: Path, method: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    artifact = source_root / "prepared" / method / "model.pt"
    metadata_path = artifact.parent / "metadata.json"
    log_path = artifact.parent / "compression_log.jsonl"
    if not artifact.exists():
        raise FileNotFoundError(f"missing prepared artifact for {method}: {artifact}")
    payload = torch.load(artifact, map_location="cpu")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"prepared artifact for {method} has no state_dict")
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "source_method": method,
            "artifact": str(artifact),
            "metadata_path": str(metadata_path),
            "compression_log": str(log_path),
            "metadata_exists": metadata_path.exists(),
            "compression_log_exists": log_path.exists(),
        }
    )
    return state, metadata


def save_original_modules(model: nn.Module, selected_names: set[str]) -> dict[str, nn.Module]:
    saved: dict[str, nn.Module] = {}
    for name in selected_names:
        parent, child_name = module_parent(model, name)
        saved[name] = getattr(parent, child_name)
    return saved


def restore_original_modules(model: nn.Module, saved: dict[str, nn.Module]) -> None:
    for name, module in saved.items():
        parent, child_name = module_parent(model, name)
        setattr(parent, child_name, module)


def install_kernel_modules(
    model: nn.Module,
    modules: list[Any],
    *,
    method: str,
    state: dict[str, torch.Tensor],
    selected_names: set[str],
    dtype: torch.dtype,
) -> dict[str, Any]:
    module_by_name = {info.name: info for info in modules}
    report: dict[str, Any] = {
        "method": method,
        "runtime_path": "selected_compressed_weight_to_real_kernel_module",
        "activation_quantization": "runtime_kernel" if method.endswith("nvfp4") else "none",
        "selected_modules": len(selected_names),
        "replaced_modules": 0,
        "skipped": [],
        "kernel_module_types": {},
    }
    for name in sorted(selected_names):
        info = module_by_name.get(name)
        if info is None:
            report["skipped"].append({"name": name, "reason": "not_in_module_list"})
            continue
        parent, child_name = module_parent(model, name)
        original = getattr(parent, child_name)
        if not isinstance(original, nn.Linear):
            report["skipped"].append({"name": name, "reason": f"not_linear:{type(original).__name__}"})
            continue
        kernel_module = build_kernel_module_from_state(original, name=name, method=method, state=state, dtype=dtype)
        setattr(parent, child_name, kernel_module)
        report["replaced_modules"] += 1
        report["kernel_module_types"][name] = type(kernel_module).__name__
    if report["skipped"]:
        raise RuntimeError(f"kernel install skipped modules: {report['skipped'][:5]}")
    return report


def build_kernel_module_from_state(
    original: nn.Linear,
    *,
    name: str,
    method: str,
    state: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> nn.Module:
    weight_key = f"{name}.weight"
    bias_key = f"{name}.bias"
    if weight_key not in state:
        raise KeyError(f"{method} artifact missing {weight_key}")
    temp = nn.Linear(
        original.in_features,
        original.out_features,
        bias=original.bias is not None,
        device=original.weight.device,
        dtype=dtype,
    )
    temp.weight.data.copy_(state[weight_key].to(device=original.weight.device, dtype=dtype))
    if temp.bias is not None:
        if bias_key in state:
            temp.bias.data.copy_(state[bias_key].to(device=original.weight.device, dtype=dtype))
        elif original.bias is not None:
            temp.bias.data.copy_(original.bias.detach().to(device=original.weight.device, dtype=dtype))
    if method == "dense_nvfp4":
        from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config, _load_cutlass_nvfp4_symbols

        nvfp4_linear_cls, can_use = _load_cutlass_nvfp4_symbols()
        if not can_use(1, temp.out_features, temp.in_features, load_extension=False):
            raise RuntimeError(f"dense_nvfp4 shape unsupported for {name}: n={temp.out_features} k={temp.in_features}")
        return nvfp4_linear_cls.from_linear(temp)
    if method == "sparse_nvfp4":
        from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config, PaddedSparseNVFP4Linear, _load_cutlass_sparse_nvfp4_symbols

        config = CutlassSparseNVFP4Config(prune=False)
        sparse_linear_cls, can_use = _load_cutlass_sparse_nvfp4_symbols()
        if not can_use(temp.out_features, config.pad_tokens_to_multiple, temp.in_features, load_extension=False):
            raise RuntimeError(f"sparse_nvfp4 shape unsupported for {name}: n={temp.out_features} k={temp.in_features}")
        return PaddedSparseNVFP4Linear(sparse_linear_cls.from_linear(temp, prune=False), config.pad_tokens_to_multiple)
    raise ValueError(method)


def local_error_metadata(
    args: Any,
    calib_metadata: dict[str, Any],
    methods: list[str],
    modules: list[Any],
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "model_key": MODEL_KEY,
        "source_root": str(args.source_root),
        "dtype": args.dtype,
        "gpu": args.gpu,
        "methods": methods,
        "modules": len(modules),
        "calibration": calib_metadata,
        "source_metadata": source_metadata,
        "validity": "kernel_aware_real_runtime_forward_with_activation_quantization",
    }


def loss_ablation_metadata(
    args: Any,
    calib_metadata: dict[str, Any],
    methods: list[str],
    policies: list[dict[str, Any]],
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "model_key": MODEL_KEY,
        "source_root": str(args.source_root),
        "dtype": args.dtype,
        "gpu": args.gpu,
        "methods": methods,
        "policies": len(policies),
        "calibration": calib_metadata,
        "metric": "mean_cross_entropy_loss_per_token",
        "source_metadata": source_metadata,
        "validity": "kernel_aware_real_runtime_forward_with_activation_quantization",
    }


__all__ = [
    "DEBUG_ROOT",
    "SOURCE_ROOT",
    "METHODS",
    "LINEAR_TYPES",
    "QualityConfig",
    "build_kernel_module_from_state",
    "cleanup_cuda",
    "compressible_modules",
    "compute_nll",
    "dtype_from_arg",
    "f",
    "install_kernel_modules",
    "layer_index",
    "load_calibration_blocks",
    "load_llama_for_quality",
    "load_prepared_payload",
    "local_cuda_index",
    "local_error_metadata",
    "loss_ablation_metadata",
    "module_family",
    "module_record",
    "module_type",
    "parse_loss_policies",
    "parse_methods",
    "quality_config_from_args",
    "read_csv",
    "read_json",
    "restore_original_modules",
    "save_original_modules",
    "select_modules",
    "weight_stats",
    "write_csv",
    "write_json",
]
