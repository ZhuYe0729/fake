#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = next(
    (parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "fake").is_dir() and (parent / "artifacts").is_dir()),
    SCRIPT_DIR.parents[4],
)
SOURCE_020_ROOT = REPO_ROOT / "artifacts/debug/020_fakevlm_uniform_accuracy"
CUTLASS_WRAPPER_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
MODELING_ROOT = CUTLASS_WRAPPER_ROOT / "modeling"
for path in (REPO_ROOT, SOURCE_020_ROOT, MODELING_ROOT, CUTLASS_WRAPPER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from eval_fakevlm_uniform_accuracy import (  # noqa: E402
    FakeVLMDataset as AccuracyDataset,
    apply_calibrated_pruning,
    replace_linear_with_dense_nvfp4_prefill_marlin_decode,
)
from fake.compression.modules import flatten_weight, restore_weight_shape, select_compressible_modules  # noqa: E402
from fake.compression.pruning import prune_dense_2_4, prune_nvfp4_pair_2_4  # noqa: E402
from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config, replace_linear_with_cutlass_nvfp4  # noqa: E402
from fake.kernels.cutlass_sparse_bf16 import (  # noqa: E402
    CutlassSparseBF16Config,
    PaddedSparseBF16Linear,
    SPARSE_BF16_BLOCKED_SHAPES,
    _load_cutlass_sparse_bf16_symbols,
    replace_linear_with_cutlass_sparse_bf16,
)
from fake.kernels.cutlass_sparse_nvfp4 import (  # noqa: E402
    CutlassSparseNVFP4Config,
    PaddedSparseNVFP4Linear,
    _load_cutlass_sparse_nvfp4_symbols,
    replace_linear_with_cutlass_sparse_nvfp4,
)
from fake.kernels.marlin_nvfp4 import MarlinNVFP4Config, replace_linear_with_marlin_nvfp4  # noqa: E402
from fake.kernels.offline_hybrid_policy import (  # noqa: E402
    DEFAULT_POLICY_KERNELS,
    LinearShapeSpec,
    ScenarioSpec,
    select_offline_hybrid_policy,
)
from fake.models.qwen3_5_kernels import QwenHybridDenseNVFP4Linear, QwenManualHybridLinear, _load_wrapper  # noqa: E402
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402


DEFAULT_MODEL_PATH = "/home/agent/wja/data/models/lingcco/fakeVLM"
DEFAULT_TEST_JSON = "/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json"
DEFAULT_IMAGE_ROOT = "/home/agent/wja/data/datasets/lingcco/FakeClue/test/test"

SCENARIOS = {
    "prefill_only": {"batch_size": 16, "input_tokens": 1024, "output_tokens": 0},
    "normal_01": {"batch_size": 1, "input_tokens": 16384, "output_tokens": 32},
    "normal_02": {"batch_size": 1, "input_tokens": 16384, "output_tokens": 256},
}

METHODS = (
    "dense_bf16",
    "uniform_dense_nvfp4",
    "uniform_sparse_bf16",
    "uniform_sparse_nvfp4",
    "uniform_marlin_weight_only",
    "uniform_dense_nvfp4_prefill_marlin_decode",
    "our_linear_hybrid",
)

UNIFORM_TO_BACKEND = {
    "uniform_dense_nvfp4": "dense_nvfp4",
    "uniform_sparse_bf16": "sparse_bf16",
    "uniform_sparse_nvfp4": "sparse_nvfp4",
    "uniform_marlin_weight_only": "marlin_nvfp4",
}


@dataclass(frozen=True)
class SpeedResult:
    prefill_ms: float
    decode_avg_ms: float
    decode_first_ms: float
    decode_steady_ms: float
    decode_total_ms: float
    e2e_ms: float


class FakeVLMSpeedDataset(Dataset):
    def __init__(
        self,
        *,
        model_path: str,
        test_json_file: str,
        image_root: str,
        sample_limit: int | None,
        input_tokens: int,
    ) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        with open(test_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data[:sample_limit] if sample_limit is not None else data
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.vision_feature_select_strategy = None
        self.input_tokens = int(input_tokens)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        image = Image.open(self.image_root / item["image"]).convert("RGB")
        inputs = self.processor(
            text=item["conversations"][0]["value"],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.input_tokens,
            truncation=True,
        )
        return {key: value.squeeze(0) for key, value in inputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FakeVLM cross-workload E2E speed task.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override-output-tokens", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{local_cuda_index(args.gpu)}")
    torch.cuda.set_device(device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    scenario = scenario_for(args)
    out_csv = args.output_root / "speed" / "e2e_speed_raw.csv"
    if not args.overwrite and has_result(out_csv, args.scenario, args.method):
        print(f"[skip] existing scenario={args.scenario} method={args.method}")
        return

    dataset = FakeVLMSpeedDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=args.sample_limit,
        input_tokens=scenario["input_tokens"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=scenario["batch_size"],
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    batch = first_batch(dataloader, device)

    calib_loader = None
    if method_needs_calibration(args.method):
        calib_dataset = AccuracyDataset(
            model_path=args.model_path,
            test_json_file=args.test_json_file,
            image_root=args.image_root,
            sample_limit=max(args.calib_samples, 1),
        )
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=args.calib_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    model = load_fakevlm(args.model_path, device)
    report = apply_method(args, model, scenario, device, calib_loader)
    result = benchmark_model(model, batch, scenario["output_tokens"], warmup=args.warmup, iters=args.iters)
    row = result_row(args, scenario, batch, report, result)
    append_csv(out_csv, [row])
    write_json(
        args.output_root / "status" / args.scenario / f"{args.method}.json",
        {"state": "done", "row": row, "finished_at": datetime.now().isoformat(timespec="seconds")},
    )
    print(f"[done] scenario={args.scenario} method={args.method} e2e_ms={result.e2e_ms:.6f}")


def scenario_for(args: argparse.Namespace) -> dict[str, int]:
    scenario = dict(SCENARIOS[args.scenario])
    if args.override_output_tokens is not None:
        scenario["output_tokens"] = int(args.override_output_tokens)
    return scenario


def load_fakevlm(model_path: str, device: torch.device) -> nn.Module:
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval().to(device)
    model.requires_grad_(False)
    return model


def apply_method(
    args: argparse.Namespace,
    model: nn.Module,
    scenario: dict[str, int],
    device: torch.device,
    calib_loader: DataLoader | None,
) -> dict[str, Any]:
    selected = select_compressible_modules(model, "fakevlm")
    selected_linears = [info for info in selected if info.kind == "linear"]
    if args.method == "dense_bf16":
        policy_path = write_simple_policy(args, scenario, selected_linears, "dense_bf16")
        return {
            "policy_json": str(policy_path),
            "replaced_linear_count": 0,
            "skipped_linear_count": 0,
            "backend_counts": {"dense_bf16": len(selected_linears)},
            "skipped": [],
        }
    if args.method in UNIFORM_TO_BACKEND:
        backend = UNIFORM_TO_BACKEND[args.method]
        policy_path = write_simple_policy(args, scenario, selected_linears, backend)
        report = apply_uniform_backend(args, model, selected, backend, device, calib_loader)
        return report_with_policy(report, policy_path)
    if args.method == "uniform_dense_nvfp4_prefill_marlin_decode":
        policy_path = write_dense_nvfp4_marlin_policy(args, scenario, selected_linears)
        report = replace_linear_with_dense_nvfp4_prefill_marlin_decode(
            model,
            decode_m_threshold=scenario["batch_size"],
        )
        return report_with_policy(report, policy_path)
    if args.method == "our_linear_hybrid":
        policy_path, policy = build_our_policy(args, scenario, selected_linears)
        report = apply_our_policy(args, model, policy, selected_linears, device, calib_loader)
        return report_with_policy(report, policy_path)
    raise ValueError(args.method)


def apply_uniform_backend(
    args: argparse.Namespace,
    model: nn.Module,
    selected: list[Any],
    backend: str,
    device: torch.device,
    calib_loader: DataLoader | None,
) -> dict[str, Any]:
    if backend == "dense_nvfp4":
        return as_report_dict(replace_linear_with_cutlass_nvfp4(model, "fakevlm", CutlassNVFP4Config()))
    if backend == "marlin_nvfp4":
        return as_report_dict(replace_linear_with_marlin_nvfp4(model, "fakevlm", MarlinNVFP4Config(activation_dtype=torch.bfloat16)))
    if calib_loader is None:
        raise RuntimeError(f"calib_loader is required for {backend}")
    prune_args = argparse.Namespace(calib_samples=args.calib_samples, calib_batch_size=args.calib_batch_size)
    if backend == "sparse_bf16":
        apply_calibrated_pruning(model, selected, calib_loader, device, prune_args, pattern="dense_2_4")
        return as_report_dict(replace_linear_with_cutlass_sparse_bf16(model, "fakevlm", CutlassSparseBF16Config(prune=False)))
    if backend == "sparse_nvfp4":
        apply_calibrated_pruning(model, selected, calib_loader, device, prune_args, pattern="nvfp4_pair_2_4")
        return as_report_dict(replace_linear_with_cutlass_sparse_nvfp4(model, "fakevlm", CutlassSparseNVFP4Config(prune=False)))
    raise ValueError(backend)


def build_our_policy(args: argparse.Namespace, scenario: dict[str, int], selected_linears: list[Any]) -> tuple[Path, dict[str, Any]]:
    policy_path = args.output_root / "policies" / args.scenario / args.method / "policy.json"
    if policy_path.exists() and not args.overwrite:
        return policy_path, read_json(policy_path)
    linears = [
        LinearShapeSpec(
            name=info.name,
            n=int(info.module.out_features),
            k=int(info.module.in_features),
            count=1,
        )
        for info in selected_linears
    ]
    predictor = KernelLatencyPredictor(model_root=args.model_root, kernels=list(DEFAULT_POLICY_KERNELS))
    offline = select_offline_hybrid_policy(
        linears,
        ScenarioSpec(
            batch_size=scenario["batch_size"],
            input_tokens=scenario["input_tokens"],
            output_tokens=scenario["output_tokens"],
        ),
        predictor,
        kernels=list(DEFAULT_POLICY_KERNELS),
        include_conversion_cost=True,
    )
    payload = {
        "policy_format": "fakevlm_cross_workload_policy_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "FakeVLM",
        "method": args.method,
        "scenario_name": args.scenario,
        "scenario": scenario,
        "modules": [module_policy_dict(row) for row in offline.modules],
    }
    write_json(policy_path, payload)
    write_csv(policy_path.with_suffix(".csv"), payload["modules"])
    return policy_path, payload


def module_policy_dict(row: Any) -> dict[str, Any]:
    payload = asdict(row)
    payload["selected_method"] = format_backend_pair(row.selected_prefill_backend, row.selected_decode_backend)
    return payload


def apply_our_policy(
    args: argparse.Namespace,
    model: nn.Module,
    policy: dict[str, Any],
    selected_linears: list[Any],
    device: torch.device,
    calib_loader: DataLoader | None,
) -> dict[str, Any]:
    selected_by_name = {info.name: info for info in selected_linears}
    sparse_names = [
        str(row["name"])
        for row in policy["modules"]
        if "sparse_" in str(row.get("selected_prefill_backend")) or "sparse_" in str(row.get("selected_decode_backend"))
    ]
    hessian: dict[str, torch.Tensor] = {}
    if sparse_names:
        if calib_loader is None:
            raise RuntimeError("calib_loader is required for sparse hybrid policy")
        from eval_fakevlm_uniform_accuracy import collect_vlm_hessian_diag

        hessian = collect_vlm_hessian_diag(
            model=model,
            modules=[selected_by_name[name] for name in sparse_names if name in selected_by_name],
            dataloader=calib_loader,
            device=device,
            input_dtype=torch.bfloat16,
            max_samples=args.calib_samples,
        )

    backend_counts: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []
    replaced = 0
    for row in policy["modules"]:
        name = str(row["name"])
        prefill_backend = optional_backend(row.get("selected_prefill_backend"))
        decode_backend = optional_backend(row.get("selected_decode_backend")) or prefill_backend
        if prefill_backend is None:
            skipped.append({"name": name, "reason": str(row.get("reason", "no_selected_backend"))})
            continue
        backend_counts[format_backend_pair(prefill_backend, decode_backend)] += 1
        if prefill_backend == decode_backend == "dense_bf16":
            continue
        try:
            parent, child_name = resolve_parent(model, name)
            linear = getattr(parent, child_name)
            if not isinstance(linear, nn.Linear):
                skipped.append({"name": name, "reason": f"not_linear:{type(linear).__name__}"})
                continue
            module = make_policy_module(
                linear,
                prefill_backend,
                decode_backend,
                decode_m_threshold=int(policy["scenario"]["batch_size"]),
                hdiag=hessian.get(name),
            )
            setattr(parent, child_name, module.eval())
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return {
        "backend": "our_linear_hybrid",
        "config": {"scenario": policy["scenario"]},
        "replaced_linear_count": replaced,
        "skipped_linear_count": len(skipped),
        "backend_counts": dict(sorted(backend_counts.items())),
        "skipped": skipped,
    }


def make_policy_module(
    linear: nn.Linear,
    prefill_backend: str,
    decode_backend: str,
    *,
    decode_m_threshold: int,
    hdiag: torch.Tensor | None,
) -> nn.Module:
    if {prefill_backend, decode_backend} == {"dense_nvfp4", "marlin_nvfp4"}:
        wrapper = _load_wrapper()
        canonical = wrapper.canonical_from_linear(linear, device=linear.weight.device)
        return QwenHybridDenseNVFP4Linear(
            canonical,
            decode_activation_dtype=torch.bfloat16,
            marlin_m_threshold=decode_m_threshold,
            prefill_backend=prefill_backend,
            decode_backend=decode_backend,
        )
    if prefill_backend == decode_backend:
        return make_backend_module(linear, prefill_backend, hdiag)
    modules = {
        prefill_backend: make_backend_module(linear, prefill_backend, hdiag),
        decode_backend: make_backend_module(linear, decode_backend, hdiag),
    }
    return QwenManualHybridLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        prefill_backend=prefill_backend,
        decode_backend=decode_backend,
        decode_m_threshold=decode_m_threshold,
        modules=modules,
    )


def make_backend_module(linear: nn.Linear, backend: str, hdiag: torch.Tensor | None) -> nn.Module:
    if backend == "dense_bf16":
        return clone_linear(linear)
    prepared = clone_linear(linear)
    if backend == "dense_nvfp4":
        nvfp4_cls, can_use = _load_cutlass_nvfp4_symbols()
        if not can_use(1, prepared.out_features, prepared.in_features, load_extension=False):
            raise ValueError(shape_reason(prepared, backend))
        return nvfp4_cls.from_linear(prepared)
    if backend == "marlin_nvfp4":
        wrapper = _load_wrapper()
        if not wrapper.can_use_marlin_nvfp4(1, prepared.out_features, prepared.in_features, load_extension=False):
            raise ValueError(shape_reason(prepared, backend))
        return wrapper.MarlinNVFP4Linear.from_linear(prepared, activation_dtype=torch.bfloat16)
    if backend == "sparse_bf16":
        if (prepared.out_features, prepared.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            raise ValueError(shape_reason(prepared, "sparse_bf16_blocked"))
        result = prune_dense_2_4(flatten_weight(prepared), hdiag)
        if result.mask is not None:
            prepared.weight.data.copy_(restore_weight_shape(prepared, result.weight))
        sparse_cls, can_use = _load_cutlass_sparse_bf16_symbols()
        if not can_use(prepared.out_features, 8, prepared.in_features, load_extension=False):
            raise ValueError(shape_reason(prepared, backend))
        return PaddedSparseBF16Linear(sparse_cls.from_linear(prepared, prune=False), 8)
    if backend == "sparse_nvfp4":
        result = prune_nvfp4_pair_2_4(flatten_weight(prepared), hdiag)
        if result.mask is not None:
            prepared.weight.data.copy_(restore_weight_shape(prepared, result.weight))
        sparse_cls, can_use = _load_cutlass_sparse_nvfp4_symbols()
        if not can_use(prepared.out_features, 32, prepared.in_features, load_extension=False):
            raise ValueError(shape_reason(prepared, backend))
        return PaddedSparseNVFP4Linear(sparse_cls.from_linear(prepared, prune=False), 32)
    raise ValueError(f"unsupported backend: {backend}")


def clone_linear(linear: nn.Linear) -> nn.Linear:
    out = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, device=linear.weight.device, dtype=torch.bfloat16)
    out.weight.data.copy_(linear.weight.detach().to(torch.bfloat16))
    if linear.bias is not None:
        out.bias.data.copy_(linear.bias.detach().to(torch.bfloat16))
    out.eval()
    out.requires_grad_(False)
    return out


@torch.inference_mode()
def benchmark_model(model: nn.Module, batch: dict[str, torch.Tensor], output_tokens: int, *, warmup: int, iters: int) -> SpeedResult:
    for _ in range(warmup):
        run_one_cycle(model, batch, min(output_tokens, 2))
    torch.cuda.synchronize()
    results = [run_one_cycle(model, batch, output_tokens) for _ in range(iters)]
    return SpeedResult(
        prefill_ms=statistics.fmean(row.prefill_ms for row in results),
        decode_avg_ms=statistics.fmean(row.decode_avg_ms for row in results),
        decode_first_ms=statistics.fmean(row.decode_first_ms for row in results),
        decode_steady_ms=statistics.fmean(row.decode_steady_ms for row in results),
        decode_total_ms=statistics.fmean(row.decode_total_ms for row in results),
        e2e_ms=statistics.fmean(row.e2e_ms for row in results),
    )


def run_one_cycle(model: nn.Module, batch: dict[str, torch.Tensor], output_tokens: int) -> SpeedResult:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    prefill_inputs = {key: value for key, value in batch.items()}
    start.record()
    out = model(**prefill_inputs, use_cache=output_tokens > 0)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = float(start.elapsed_time(end))
    logits = out.logits
    assert_finite(logits[:, -1:])
    if output_tokens <= 0:
        del out, logits
        return SpeedResult(prefill_ms, 0.0, 0.0, 0.0, 0.0, prefill_ms)

    past_key_values = out.past_key_values
    next_token = logits[:, -1:].argmax(dim=-1)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.clone()
    times = []
    for _ in range(output_tokens):
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
        decode_inputs = {
            "input_ids": next_token,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        if attention_mask is not None:
            decode_inputs["attention_mask"] = attention_mask
        start.record()
        out = model(**decode_inputs)
        end.record()
        torch.cuda.synchronize()
        step_ms = float(start.elapsed_time(end))
        logits = out.logits
        assert_finite(logits[:, -1:])
        times.append(step_ms)
        past_key_values = out.past_key_values
        next_token = logits[:, -1:].argmax(dim=-1)
    decode_total = float(sum(times))
    decode_avg = decode_total / len(times)
    decode_steady = sum(times[1:]) / max(len(times) - 1, 1)
    return SpeedResult(prefill_ms, decode_avg, times[0], decode_steady, decode_total, prefill_ms + decode_total)


def first_batch(dataloader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError("dataset is empty") from exc
    return move_inputs(batch, device)


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


def write_simple_policy(args: argparse.Namespace, scenario: dict[str, int], selected_linears: list[Any], backend: str) -> Path:
    policy_path = args.output_root / "policies" / args.scenario / args.method / "policy.json"
    modules = [
        {
            "name": info.name,
            "n": int(info.module.out_features),
            "k": int(info.module.in_features),
            "selected_prefill_backend": backend,
            "selected_decode_backend": backend if scenario["output_tokens"] > 0 else "",
            "selected_method": backend,
        }
        for info in selected_linears
    ]
    payload = base_policy_payload(args, scenario, modules)
    write_json(policy_path, payload)
    write_csv(policy_path.with_suffix(".csv"), modules)
    return policy_path


def write_dense_nvfp4_marlin_policy(args: argparse.Namespace, scenario: dict[str, int], selected_linears: list[Any]) -> Path:
    policy_path = args.output_root / "policies" / args.scenario / args.method / "policy.json"
    modules = [
        {
            "name": info.name,
            "n": int(info.module.out_features),
            "k": int(info.module.in_features),
            "selected_prefill_backend": "dense_nvfp4",
            "selected_decode_backend": "marlin_nvfp4" if scenario["output_tokens"] > 0 else "",
            "selected_method": "dense_nvfp4->marlin_nvfp4" if scenario["output_tokens"] > 0 else "dense_nvfp4",
        }
        for info in selected_linears
    ]
    payload = base_policy_payload(args, scenario, modules)
    write_json(policy_path, payload)
    write_csv(policy_path.with_suffix(".csv"), modules)
    return policy_path


def base_policy_payload(args: argparse.Namespace, scenario: dict[str, int], modules: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policy_format": "fakevlm_cross_workload_policy_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "FakeVLM",
        "method": args.method,
        "scenario_name": args.scenario,
        "scenario": scenario,
        "modules": modules,
    }


def result_row(
    args: argparse.Namespace,
    scenario: dict[str, int],
    batch: dict[str, torch.Tensor],
    report: dict[str, Any],
    result: SpeedResult,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario": args.scenario,
        "method": args.method,
        "batch_size": scenario["batch_size"],
        "actual_batch_size": int(batch["input_ids"].shape[0]),
        "input_tokens": scenario["input_tokens"],
        "actual_input_tokens": int(batch["input_ids"].shape[1]),
        "output_tokens": scenario["output_tokens"],
        "warmup": args.warmup,
        "iters": args.iters,
        "prefill_ms": f"{result.prefill_ms:.6f}",
        "decode_avg_ms": f"{result.decode_avg_ms:.6f}",
        "decode_first_ms": f"{result.decode_first_ms:.6f}",
        "decode_steady_ms": f"{result.decode_steady_ms:.6f}",
        "decode_total_ms": f"{result.decode_total_ms:.6f}",
        "e2e_ms": f"{result.e2e_ms:.6f}",
        "samples_per_sec": f"{scenario['batch_size'] * 1000.0 / result.e2e_ms:.6f}" if result.e2e_ms > 0 else "0.000000",
        "policy_json": report.get("policy_json", ""),
        "replaced_linear_count": report.get("replaced_linear_count", ""),
        "skipped_linear_count": report.get("skipped_linear_count", ""),
        "backend_counts": json.dumps(report.get("backend_counts", {}), sort_keys=True),
        "skipped": json.dumps(report.get("skipped", [])[:20], sort_keys=True),
        "calib_samples": args.calib_samples if method_needs_calibration(args.method) else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(local_cuda_index(args.gpu)) if torch.cuda.is_available() else "",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def method_needs_calibration(method: str) -> bool:
    return method in {"uniform_sparse_bf16", "uniform_sparse_nvfp4", "our_linear_hybrid"}


def report_with_policy(report: Any, policy_path: Path) -> dict[str, Any]:
    payload = as_report_dict(report)
    payload["policy_json"] = str(policy_path)
    return payload


def as_report_dict(report: Any) -> dict[str, Any]:
    if isinstance(report, dict):
        payload = dict(report)
    elif hasattr(report, "__dataclass_fields__"):
        payload = asdict(report)
    else:
        payload = dict(report.__dict__)
    if "backend_counts" not in payload:
        backend = str(payload.get("backend", payload.get("replacement_backend", "")))
        replaced = int(payload.get("replaced_linear_count", 0) or 0)
        skipped = int(payload.get("skipped_linear_count", 0) or 0)
        payload["backend_counts"] = {backend: replaced, "skipped": skipped}
    return payload


def _load_cutlass_nvfp4_symbols() -> tuple[type[nn.Module], Any]:
    wrapper = _load_wrapper()
    return wrapper.NVFP4Linear, wrapper.can_use_cutlass_nvfp4


def resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def optional_backend(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "None", "null"} else text


def format_backend_pair(prefill: str | None, decode: str | None) -> str:
    if decode is None or decode == "" or decode == prefill:
        return str(prefill)
    return f"{prefill}->{decode}"


def shape_reason(linear: nn.Linear, backend: str) -> str:
    return f"shape_not_supported:{backend}:N={linear.out_features},K={linear.in_features}"


def assert_finite(tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all().item():
        raise RuntimeError("non-finite tensor encountered")


def has_result(path: Path, scenario: str, method: str) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scenario") == scenario and row.get("method") == method:
                return True
    return False


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        fcntl.flock(lock, fcntl.LOCK_UN)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_cuda_index(requested_gpu: int) -> int:
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA is required")
    if requested_gpu < count:
        return requested_gpu
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return 0
    raise RuntimeError(f"requested gpu {requested_gpu}, but torch sees {count} CUDA devices")


if __name__ == "__main__":
    main()
