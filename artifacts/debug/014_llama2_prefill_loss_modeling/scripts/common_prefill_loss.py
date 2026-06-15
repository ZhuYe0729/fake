#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


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
    apply_compressed_weights,
    cleanup_cuda,
    compressible_modules,
    compute_nll,
    dtype_from_arg,
    layer_bucket,
    layer_index,
    load_calibration_blocks,
    load_llama_for_quality,
    load_prepared_state,
    local_cuda_index,
    module_family,
    module_record,
    module_type,
    parse_policy,
    weight_stats,
)
from common import module_parent, utc_now  # type: ignore  # noqa: E402


METHODS = ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def save_original_weights(model: torch.nn.Module, selected_names: set[str]) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    saved: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for name in selected_names:
        parent, child_name = module_parent(model, name)
        module = getattr(parent, child_name)
        saved[name] = (
            module.weight.detach().cpu().clone(),
            module.bias.detach().cpu().clone() if module.bias is not None else None,
        )
    return saved


def restore_original_weights(model: torch.nn.Module, saved: dict[str, tuple[torch.Tensor, torch.Tensor | None]]) -> None:
    for name, (weight, bias) in saved.items():
        parent, child_name = module_parent(model, name)
        module = getattr(parent, child_name)
        module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
        if module.bias is not None and bias is not None:
            module.bias.data.copy_(bias.to(device=module.bias.device, dtype=module.bias.dtype))


def local_error_metadata(args: Any, calib_metadata: dict[str, Any], methods: list[str], modules: list[Any]) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "model_key": MODEL_KEY,
        "source_root": str(args.source_root),
        "dtype": args.dtype,
        "gpu": args.gpu,
        "methods": methods,
        "modules": len(modules),
        "calibration": calib_metadata,
    }


def loss_ablation_metadata(args: Any, calib_metadata: dict[str, Any], methods: list[str], policies: list[dict[str, Any]]) -> dict[str, Any]:
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
    }


__all__ = [
    "DEBUG_ROOT",
    "SOURCE_ROOT",
    "METHODS",
    "LINEAR_TYPES",
    "QualityConfig",
    "apply_compressed_weights",
    "cleanup_cuda",
    "compressible_modules",
    "compute_nll",
    "dtype_from_arg",
    "f",
    "layer_bucket",
    "layer_index",
    "load_calibration_blocks",
    "load_llama_for_quality",
    "load_prepared_state",
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
    "restore_original_weights",
    "save_original_weights",
    "select_modules",
    "weight_stats",
    "write_csv",
    "write_json",
]
