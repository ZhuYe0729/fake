#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common_fakevlm_pareto import (
    DEBUG_ROOT,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_JSON,
    METHODS,
    f,
    layer_bucket,
    layer_index,
    local_cuda_index,
    module_family,
    module_type,
    parse_methods,
    write_csv,
    write_json,
)

from fake.compression.modules import flatten_weight, restore_weight_shape, select_compressible_modules
from fake.compression.pruning import prune_dense_2_4, prune_nvfp4_pair_2_4
from fake.kernels.cutlass_nvfp4 import _load_cutlass_nvfp4_symbols
from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear, SPARSE_BF16_BLOCKED_SHAPES, _load_cutlass_sparse_bf16_symbols
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear, _load_cutlass_sparse_nvfp4_symbols


@dataclass(frozen=True)
class LinearSpec:
    name: str
    n: int
    k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect FakeVLM per-module local output errors for Pareto quality modeling.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--module-chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from eval_fakevlm_uniform_accuracy import FakeVLMDataset, collect_vlm_hessian_diag
    from run_fakevlm_prefill_speed import load_fakevlm

    set_seed(args.seed)
    methods = parse_methods(args.methods)
    output_path = args.output_root / "sensitivity" / "module_method_local_errors.csv"
    feature_path = args.output_root / "sensitivity" / "module_features.csv"
    if output_path.exists() and not args.overwrite:
        print(f"skip existing local errors: {output_path}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{local_cuda_index(args.gpu)}")
    torch.cuda.set_device(device)

    dataset = FakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=args.sample_limit,
    )
    calib_dataset = FakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=max(args.calib_samples, 1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    calib_loader = DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = load_fakevlm(args.model_path, device)
    selected_infos = [info for info in select_compressible_modules(model, "fakevlm") if info.kind == "linear"]
    if args.max_modules is not None:
        selected_infos = selected_infos[: args.max_modules]
    modules = [LinearSpec(info.name, int(info.module.out_features), int(info.module.in_features)) for info in selected_infos]
    if not modules:
        raise RuntimeError("No FakeVLM language linear modules selected")
    write_csv(feature_path, [module_feature_row(i, spec) for i, spec in enumerate(modules, start=1)])

    hessian = {}
    if any(method.startswith("sparse_") for method in methods):
        hessian = collect_vlm_hessian_diag(
            model=model,
            modules=selected_infos,
            dataloader=calib_loader,
            device=device,
            input_dtype=torch.bfloat16,
            max_samples=args.calib_samples,
        )

    inputs_by_module = collect_module_inputs(model, selected_infos, loader, device, max_samples=args.sample_limit)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(modules), args.module_chunk_size):
        for module_index, spec in enumerate(modules[start : start + args.module_chunk_size], start=start + 1):
            linear = get_module(model, spec.name)
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(f"expected Linear for {spec.name}, got {type(linear).__name__}")
            x = inputs_by_module.get(spec.name)
            if x is None:
                raise RuntimeError(f"no captured input for module {spec.name}")
            ref = linear(x).detach()
            for method in methods:
                row = module_feature_row(module_index, spec)
                row["method"] = method
                if method == "dense_bf16":
                    row.update(zero_error())
                    rows.append(row)
                    continue
                candidate_linear = clone_linear_for_method(linear, method, hessian.get(spec.name))
                backend = make_backend_module_from_prepared(method, candidate_linear).to(device).eval()
                with torch.inference_mode():
                    out = backend(x)
                    assert_finite(out)
                row.update(error_metrics(ref, out))
                rows.append(row)
                del backend, candidate_linear, out
                gc.collect()
                torch.cuda.empty_cache()
        write_csv(output_path, rows)
        print(f"wrote {len(rows)} rows to {output_path}")

    write_json(
        args.output_root / "sensitivity" / "collect_local_errors_metadata.json",
        {
            "model_path": args.model_path,
            "test_json_file": args.test_json_file,
            "image_root": args.image_root,
            "methods": methods,
            "selected_modules": len(modules),
            "sample_limit": args.sample_limit,
            "calib_samples": args.calib_samples,
            "batch_size": args.batch_size,
        },
    )


def collect_module_inputs(model: nn.Module, selected_infos: list[Any], loader: DataLoader, device: torch.device, *, max_samples: int) -> dict[str, torch.Tensor]:
    from eval_fakevlm_uniform_accuracy import move_inputs

    captured: dict[str, list[torch.Tensor]] = {info.name: [] for info in selected_infos}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            if sum(t.shape[0] for t in captured[name]) >= max_samples:
                return
            x = inputs[0].detach().to(dtype=torch.bfloat16)
            captured[name].append(x)

        return hook

    for info in selected_infos:
        handles.append(info.module.register_forward_hook(make_hook(info.name)))
    processed = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = move_inputs(batch["inputs"], device, torch.bfloat16)
                current = int(next(iter(inputs.values())).shape[0])
                model(**inputs, use_cache=False)
                processed += current
                if processed >= max_samples:
                    break
    finally:
        for handle in handles:
            handle.remove()
    out = {}
    for name, chunks in captured.items():
        if chunks:
            out[name] = torch.cat(chunks, dim=0)[:max_samples].contiguous()
    return out


def clone_linear_for_method(linear: nn.Linear, method: str, hdiag: torch.Tensor | None) -> nn.Linear:
    cloned = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, device=linear.weight.device, dtype=torch.bfloat16)
    cloned.weight.data.copy_(linear.weight.detach().to(dtype=torch.bfloat16))
    if linear.bias is not None:
        cloned.bias.data.copy_(linear.bias.detach().to(dtype=torch.bfloat16))
    cloned.eval()
    cloned.requires_grad_(False)
    if method == "sparse_bf16":
        result = prune_dense_2_4(flatten_weight(cloned), hdiag)
    elif method == "sparse_nvfp4":
        result = prune_nvfp4_pair_2_4(flatten_weight(cloned), hdiag)
    else:
        result = None
    if result is not None and result.mask is not None:
        cloned.weight.data.copy_(restore_weight_shape(cloned, result.weight))
    return cloned


def make_backend_module_from_prepared(method: str, linear: nn.Linear) -> nn.Module:
    if method == "dense_nvfp4":
        nvfp4_cls, can_use = _load_cutlass_nvfp4_symbols()
        if not can_use(1, linear.out_features, linear.in_features, load_extension=False):
            raise ValueError(f"shape_not_supported:dense_nvfp4:{linear.out_features}x{linear.in_features}")
        return nvfp4_cls.from_linear(linear)
    if method == "sparse_bf16":
        if (linear.out_features, linear.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            raise ValueError(f"shape_not_supported:sparse_bf16_blocked:{linear.out_features}x{linear.in_features}")
        sparse_cls, can_use = _load_cutlass_sparse_bf16_symbols()
        if not can_use(linear.out_features, 8, linear.in_features, load_extension=False):
            raise ValueError(f"shape_not_supported:sparse_bf16:{linear.out_features}x{linear.in_features}")
        return PaddedSparseBF16Linear(sparse_cls.from_linear(linear, prune=False), 8)
    if method == "sparse_nvfp4":
        sparse_cls, can_use = _load_cutlass_sparse_nvfp4_symbols()
        if not can_use(linear.out_features, 32, linear.in_features, load_extension=False):
            raise ValueError(f"shape_not_supported:sparse_nvfp4:{linear.out_features}x{linear.in_features}")
        return PaddedSparseNVFP4Linear(sparse_cls.from_linear(linear, prune=False), 32)
    raise ValueError(f"unsupported backend for local error: {method}")


def assert_finite(tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all().item():
        raise RuntimeError("output contains NaN/Inf")


def module_feature_row(index: int, spec: LinearSpec) -> dict[str, Any]:
    layer = layer_index(spec.name)
    return {
        "module_index": index,
        "module_name": spec.name,
        "layer": layer,
        "layer_bucket": layer_bucket(layer),
        "module_type": module_type(spec.name),
        "module_family": module_family(spec.name),
        "out_features": spec.n,
        "in_features": spec.k,
        "numel": spec.n * spec.k,
    }


def zero_error() -> dict[str, float]:
    return {
        "output_mse": 0.0,
        "output_rel_mse": 0.0,
        "output_rmse_over_rms": 0.0,
        "output_max_abs_error": 0.0,
        "output_ref_rms": 0.0,
    }


def error_metrics(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    ref_f = ref.float()
    out_f = out.float()
    diff = out_f - ref_f
    mse = diff.pow(2).mean().item()
    ref_power = ref_f.pow(2).mean().item()
    return {
        "output_mse": mse,
        "output_rel_mse": mse / max(ref_power, 1e-12),
        "output_rmse_over_rms": (mse / max(ref_power, 1e-12)) ** 0.5,
        "output_max_abs_error": diff.abs().max().item(),
        "output_ref_rms": ref_power ** 0.5,
    }


def get_module(model: nn.Module, name: str) -> nn.Module:
    module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
