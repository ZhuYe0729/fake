#!/usr/bin/env python
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in (CODE_DIR, *CODE_DIR.parents) if (parent / "fake").is_dir() and (parent / "artifacts").is_dir()),
    CODE_DIR.parents[3],
)
CUTLASS_WRAPPER_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
KERNEL_PREDICTOR_PATHS = list(REPO_ROOT.glob("fake/**/modeling/kernel_predictor.py"))
MODELING_ROOT = KERNEL_PREDICTOR_PATHS[0].parents[1] if KERNEL_PREDICTOR_PATHS else CUTLASS_WRAPPER_ROOT
for path in (CODE_DIR, REPO_ROOT, CUTLASS_WRAPPER_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from fake.compression.modules import select_compressible_modules  # noqa: E402
from fake.kernels.cutlass_nvfp4 import _load_cutlass_nvfp4_symbols  # noqa: E402
from fake.kernels.cutlass_sparse_bf16 import (  # noqa: E402
    PaddedSparseBF16Linear,
    SPARSE_BF16_BLOCKED_SHAPES,
    _load_cutlass_sparse_bf16_symbols,
)
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear, _load_cutlass_sparse_nvfp4_symbols  # noqa: E402
try:
    from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402
except ModuleNotFoundError as exc:
    searched = "\n".join(str(path) for path in (CUTLASS_WRAPPER_ROOT, MODELING_ROOT, *KERNEL_PREDICTOR_PATHS))
    raise ModuleNotFoundError(
        "Could not import modeling.kernel_predictor. "
        "Expected fake/kernels/cutlass/cutlass_wrapper/modeling/kernel_predictor.py. "
        f"Searched:\n{searched}"
    ) from exc


DEFAULT_MODEL_PATH = "/home/agent/wja/data/models/lingcco/fakeVLM"
DEFAULT_TEST_JSON = "/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json"
DEFAULT_IMAGE_ROOT = "/home/agent/wja/data/datasets/lingcco/FakeClue/test/test"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed"
BACKENDS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
POLICY_FAMILIES = ("manual_profile", "latency_model")
UNIFORM_FAMILIES = tuple(f"uniform_{backend}" for backend in BACKENDS)
POLICY_FORMAT = "fakevlm_prefill_linear_policy_v1"


@dataclass(frozen=True)
class LinearSpec:
    name: str
    n: int
    k: int


@dataclass(frozen=True)
class SpeedResult:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float


class FakeVLMDataset(Dataset):
    def __init__(
        self,
        *,
        model_path: str,
        test_json_file: str,
        image_root: str,
        sample_limit: int | None,
    ) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        with open(test_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data[:sample_limit] if sample_limit is not None else data
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.vision_feature_select_strategy = None

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
            max_length=1024,
            truncation=True,
        )
        return {key: value.squeeze(0) for key, value in inputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FakeVLM prefill-only per-linear hybrid speed.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--manual-warmup", type=int, default=3)
    parser.add_argument("--manual-iters", type=int, default=10)
    parser.add_argument("--families", nargs="+", default=[*POLICY_FAMILIES, *UNIFORM_FAMILIES])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FakeVLM speed benchmarking.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_static_docs(args.output_root)
    write_json(args.output_root / "configs" / f"batch_{args.batch_size}_run_config.json", run_config(args))

    device = torch.device(args.device)
    dataset = FakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=args.sample_limit,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    batch = first_batch(dataloader, device=device)
    actual_batch_size = int(batch["input_ids"].shape[0])
    input_tokens = int(batch["input_ids"].shape[1])
    m_prefill = actual_batch_size * input_tokens

    template_model = load_fakevlm(args.model_path, device)
    linears = enumerate_fakevlm_linears(template_model)
    del template_model
    gc.collect()
    torch.cuda.empty_cache()

    policies: dict[str, Path] = {}
    if "manual_profile" in args.families:
        policies["manual_profile"] = build_manual_policy(args, linears, m_prefill, actual_batch_size, input_tokens, device)
    if "latency_model" in args.families:
        policies["latency_model"] = build_latency_model_policy(args, linears, m_prefill, actual_batch_size, input_tokens)
    for family in args.families:
        if family.startswith("uniform_"):
            backend = family.removeprefix("uniform_")
            if backend not in BACKENDS:
                raise ValueError(f"Unsupported uniform family: {family}")
            policies[family] = build_uniform_policy(args, linears, m_prefill, actual_batch_size, input_tokens, backend)

    for family in args.families:
        policy_path = policies.get(family)
        if policy_path is None:
            continue
        speed_path = args.output_root / "speed" / "prefill_speed.csv"
        if not args.overwrite and has_speed_row(speed_path, family, args.batch_size):
            print(f"[skip] existing speed row family={family} batch={args.batch_size}")
            continue
        model = load_fakevlm(args.model_path, device)
        policy = load_policy(policy_path)
        report = apply_policy(model, policy)
        result = benchmark_prefill(model, batch, warmup=args.warmup, iters=args.iters)
        row = speed_row(args, family, policy_path, policy, report, result, batch)
        append_csv_row(speed_path, row)
        print(
            f"[done] family={family} batch={args.batch_size} "
            f"mean_ms={result.mean_ms:.6f} samples_per_sec={row['samples_per_sec']} backends={row['backend_counts']}"
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()


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


def enumerate_fakevlm_linears(model: nn.Module) -> list[LinearSpec]:
    specs = []
    for info in select_compressible_modules(model, "fakevlm"):
        if info.kind != "linear":
            continue
        specs.append(LinearSpec(info.name, int(info.module.out_features), int(info.module.in_features)))
    if not specs:
        raise RuntimeError("No FakeVLM language linear modules were selected.")
    return specs


def build_manual_policy(
    args: argparse.Namespace,
    linears: list[LinearSpec],
    m_prefill: int,
    actual_batch_size: int,
    input_tokens: int,
    device: torch.device,
) -> Path:
    policy_path = policy_path_for(args.output_root, "manual_profile", args.batch_size)
    if policy_path.exists() and not args.overwrite:
        return policy_path
    candidates_by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for index, shape in enumerate(sorted({(spec.n, spec.k) for spec in linears})):
        n, k = shape
        shape_rows = []
        for backend in BACKENDS:
            row = benchmark_backend(
                backend=backend,
                m=m_prefill,
                n=n,
                k=k,
                device=device,
                warmup=args.manual_warmup,
                iters=args.manual_iters,
                seed=args.seed + 1000 + index,
            )
            row.update({"batch_size": args.batch_size, "m": m_prefill, "n": n, "k": k, "backend": backend})
            rows.append(row)
            shape_rows.append(row)
        candidates_by_shape[shape] = shape_rows
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(args.output_root / "candidates" / "manual_profile" / f"batch_{args.batch_size}.csv", rows)
    modules = []
    for spec in linears:
        selected = select_best_candidate(candidates_by_shape[(spec.n, spec.k)])
        modules.append(policy_module(spec, selected["backend"], selected.get("latency_ms"), selected.get("reason", "")))
    write_policy(args, "manual_profile", m_prefill, actual_batch_size, input_tokens, modules)
    return policy_path


def build_latency_model_policy(
    args: argparse.Namespace,
    linears: list[LinearSpec],
    m_prefill: int,
    actual_batch_size: int,
    input_tokens: int,
) -> Path:
    policy_path = policy_path_for(args.output_root, "latency_model", args.batch_size)
    if policy_path.exists() and not args.overwrite:
        return policy_path
    predictor = KernelLatencyPredictor(model_root=args.model_root, kernels=list(BACKENDS))
    candidate_rows = []
    by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for n, k in sorted({(spec.n, spec.k) for spec in linears}):
        shape_rows = []
        for backend in BACKENDS:
            pred_m = prediction_m_for_backend(m_prefill, backend)
            row = predict_backend(predictor, backend, pred_m, n, k)
            row.update({"batch_size": args.batch_size, "m": m_prefill, "prediction_m": pred_m, "n": n, "k": k, "backend": backend})
            candidate_rows.append(row)
            shape_rows.append(row)
        by_shape[(n, k)] = shape_rows
    write_csv(args.output_root / "candidates" / "latency_model" / f"batch_{args.batch_size}.csv", candidate_rows)
    modules = []
    for spec in linears:
        selected = select_best_candidate(by_shape[(spec.n, spec.k)])
        modules.append(policy_module(spec, selected["backend"], selected.get("latency_ms"), selected.get("reason", "")))
    write_policy(args, "latency_model", m_prefill, actual_batch_size, input_tokens, modules)
    return policy_path


def build_uniform_policy(
    args: argparse.Namespace,
    linears: list[LinearSpec],
    m_prefill: int,
    actual_batch_size: int,
    input_tokens: int,
    backend: str,
) -> Path:
    family = f"uniform_{backend}"
    policy_path = policy_path_for(args.output_root, family, args.batch_size)
    if policy_path.exists() and not args.overwrite:
        return policy_path
    modules = [policy_module(spec, backend, None, "") for spec in linears]
    write_policy(args, family, m_prefill, actual_batch_size, input_tokens, modules)
    return policy_path


def benchmark_backend(
    *,
    backend: str,
    m: int,
    n: int,
    k: int,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    base = make_base_linear(n, k, device, seed)
    try:
        module = make_backend_module(backend, base).eval()
        x = torch.randn((1, m, k), device=device, dtype=torch.bfloat16)
        for _ in range(warmup):
            assert_finite(module(x))
        torch.cuda.synchronize()
        result = time_cuda(lambda: module(x), iters)
        return {
            "supported": True,
            "reason": "",
            "latency_ms": result.mean_ms,
            "latency_p50_ms": result.p50_ms,
            "latency_p90_ms": result.p90_ms,
            "latency_min_ms": result.min_ms,
            "latency_max_ms": result.max_ms,
            "source": "manual_profile",
        }
    except Exception as exc:
        return {
            "supported": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "latency_ms": "",
            "latency_p50_ms": "",
            "latency_p90_ms": "",
            "latency_min_ms": "",
            "latency_max_ms": "",
            "source": "manual_profile",
        }
    finally:
        del base


def predict_backend(predictor: Any, backend: str, m: int, n: int, k: int) -> dict[str, Any]:
    selection = predictor.predict(m, n, k)
    record = next((candidate for candidate in selection.candidates if candidate.kernel == backend), None)
    if record is None:
        return {"supported": False, "reason": f"missing_prediction:{backend}", "latency_ms": "", "source": "latency_model"}
    return {
        "supported": bool(record.supported),
        "reason": record.reason,
        "latency_ms": "" if record.latency_ms is None else float(record.latency_ms),
        "source": record.source,
        "prediction_status": record.prediction_status,
        "prediction_error": record.prediction_error,
    }


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [row for row in rows if bool(row.get("supported")) and row.get("latency_ms") != ""]
    if not viable:
        return {"backend": "dense_bf16", "latency_ms": None, "reason": "fallback_dense_bf16_no_supported_candidate"}
    return min(viable, key=lambda row: float(row["latency_ms"]))


def make_base_linear(n: int, k: int, device: torch.device, seed: int) -> nn.Linear:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    linear = nn.Linear(k, n, bias=False, device=device, dtype=torch.bfloat16)
    linear.weight.data.normal_(mean=0.0, std=0.02, generator=generator)
    linear.eval()
    linear.requires_grad_(False)
    return linear


def make_backend_module(backend: str, base: nn.Linear) -> nn.Module:
    if backend == "dense_bf16":
        return clone_linear(base)
    if backend == "dense_nvfp4":
        nvfp4_cls, can_use = _load_cutlass_nvfp4_symbols()
        if not can_use(1, base.out_features, base.in_features, load_extension=False):
            raise ValueError(shape_reason(base, backend))
        return nvfp4_cls.from_linear(base)
    if backend == "sparse_bf16":
        if (base.out_features, base.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            raise ValueError(shape_reason(base, "sparse_bf16_blocked"))
        sparse_cls, can_use = _load_cutlass_sparse_bf16_symbols()
        if not can_use(base.out_features, 8, base.in_features, load_extension=False):
            raise ValueError(shape_reason(base, backend))
        return PaddedSparseBF16Linear(sparse_cls.from_linear(base, prune=True), 8)
    if backend == "sparse_nvfp4":
        sparse_cls, can_use = _load_cutlass_sparse_nvfp4_symbols()
        if not can_use(base.out_features, 32, base.in_features, load_extension=False):
            raise ValueError(shape_reason(base, backend))
        return PaddedSparseNVFP4Linear(sparse_cls.from_linear(base, prune=True), 32)
    raise ValueError(f"Unsupported backend: {backend}")


def apply_policy(model: nn.Module, policy: dict[str, Any]) -> dict[str, Any]:
    skipped: list[dict[str, str]] = []
    backend_counts: Counter[str] = Counter()
    replaced = 0
    for item in policy["modules"]:
        name = str(item["name"])
        backend = str(item["backend"])
        try:
            parent, child_name = resolve_parent(model, name)
            linear = getattr(parent, child_name)
            if not isinstance(linear, nn.Linear):
                skipped.append({"name": name, "reason": f"not_linear:{type(linear).__name__}"})
                continue
            if backend == "dense_bf16":
                backend_counts[backend] += 1
                continue
            module = make_backend_module(backend, linear)
            setattr(parent, child_name, module.eval())
            backend_counts[backend] += 1
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return {
        "replaced_linear_count": replaced,
        "skipped_linear_count": len(skipped),
        "skipped": skipped,
        "backend_counts": dict(backend_counts),
    }


@torch.inference_mode()
def benchmark_prefill(model: nn.Module, batch: dict[str, torch.Tensor], *, warmup: int, iters: int) -> SpeedResult:
    for _ in range(warmup):
        out = model(**batch, use_cache=False)
        assert_finite(out.logits)
    torch.cuda.synchronize()
    return time_cuda(lambda: model(**batch, use_cache=False).logits, iters)


def time_cuda(fn: Callable[[], torch.Tensor], iters: int) -> SpeedResult:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        assert_finite(result)
        times.append(float(start.elapsed_time(end)))
        del result
    mean_ms = statistics.fmean(times)
    return SpeedResult(
        mean_ms=mean_ms,
        p50_ms=statistics.median(times),
        p90_ms=percentile(times, 90),
        min_ms=min(times),
        max_ms=max(times),
    )


def first_batch(dataloader: DataLoader, *, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError("FakeVLM dataset is empty.") from exc
    return move_inputs(batch, device)


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


def policy_module(spec: LinearSpec, backend: str, latency_ms: Any, reason: str) -> dict[str, Any]:
    return {
        "name": spec.name,
        "backend": backend,
        "n": spec.n,
        "k": spec.k,
        "selected_prefill_latency_ms": latency_ms,
        "reason": reason,
    }


def write_policy(
    args: argparse.Namespace,
    family: str,
    m_prefill: int,
    actual_batch_size: int,
    input_tokens: int,
    modules: list[dict[str, Any]],
) -> None:
    path = policy_path_for(args.output_root, family, args.batch_size)
    payload = {
        "policy_format": POLICY_FORMAT,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "FakeVLM",
        "family": family,
        "scenario": {
            "mode": "prefill_only",
            "batch_size": args.batch_size,
            "actual_batch_size": actual_batch_size,
            "m_prefill": m_prefill,
            "input_tokens": input_tokens,
        },
        "backends": list(BACKENDS),
        "modules": modules,
    }
    write_json(path, payload)
    write_csv(path.with_suffix(".csv"), modules)


def load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("policy_format") != POLICY_FORMAT:
        raise ValueError(f"Unsupported policy format: {payload.get('policy_format')}")
    return payload


def policy_path_for(output_root: Path, family: str, batch_size: int) -> Path:
    return output_root / "policies" / family / f"batch_{batch_size}" / "policy.json"


def speed_row(
    args: argparse.Namespace,
    family: str,
    policy_path: Path,
    policy: dict[str, Any],
    report: dict[str, Any],
    result: SpeedResult,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    batch_size = int(batch["input_ids"].shape[0])
    input_tokens = int(batch["input_ids"].shape[1])
    samples_per_sec = batch_size * 1000.0 / result.mean_ms if result.mean_ms > 0 else 0.0
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "FakeVLM",
        "scenario": "prefill_only",
        "family": family,
        "batch_size": args.batch_size,
        "actual_batch_size": batch_size,
        "input_tokens": input_tokens,
        "m_prefill": batch_size * input_tokens,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_mean_ms": f"{result.mean_ms:.6f}",
        "latency_p50_ms": f"{result.p50_ms:.6f}",
        "latency_p90_ms": f"{result.p90_ms:.6f}",
        "latency_min_ms": f"{result.min_ms:.6f}",
        "latency_max_ms": f"{result.max_ms:.6f}",
        "samples_per_sec": f"{samples_per_sec:.6f}",
        "policy_path": str(policy_path),
        "selected_linear_count": len(policy["modules"]),
        "replaced_linear_count": report["replaced_linear_count"],
        "skipped_linear_count": report["skipped_linear_count"],
        "backend_counts": json.dumps(report["backend_counts"], sort_keys=True),
        "skipped": json.dumps(report["skipped"][:20], sort_keys=True),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def has_speed_row(path: Path, family: str, batch_size: int) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("family") == family and row.get("batch_size") == str(batch_size):
                return True
    return False


def prediction_m_for_backend(m: int, backend: str) -> int:
    if backend == "sparse_nvfp4":
        return round_up(m, 32)
    if backend == "sparse_bf16":
        return round_up(m, 8)
    return int(m)


def clone_linear(linear: nn.Linear) -> nn.Linear:
    out = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, device=linear.weight.device, dtype=torch.bfloat16)
    out.weight.data.copy_(linear.weight.detach().to(dtype=torch.bfloat16))
    if linear.bias is not None:
        out.bias.data.copy_(linear.bias.detach().to(dtype=torch.bfloat16))
    out.eval()
    out.requires_grad_(False)
    return out


def assert_finite(tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all().item():
        raise RuntimeError("output contains NaN/Inf")


def resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def shape_reason(linear: nn.Linear, backend: str) -> str:
    return f"shape_not_supported:{backend}:in_features={linear.in_features},out_features={linear.out_features}"


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path,
        "test_json_file": args.test_json_file,
        "image_root": args.image_root,
        "batch_size": args.batch_size,
        "sample_limit": args.sample_limit,
        "workers": args.workers,
        "warmup": args.warmup,
        "iters": args.iters,
        "manual_warmup": args.manual_warmup,
        "manual_iters": args.manual_iters,
        "families": args.families,
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def write_static_docs(output_root: Path) -> None:
    todo = output_root / "TODO_prefill_decode.md"
    if not todo.exists():
        todo.write_text(
            "# TODO: FakeVLM Prefill-Decode Hybrid Speed\n\n"
            "This run is prefill-only. Future work should add cached decode timing, decode backend selection, "
            "and conversion costs for mixed prefill/decode backend pairs.\n"
        )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        fcntl.flock(lock, fcntl.LOCK_UN)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
