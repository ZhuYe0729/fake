#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "fake").is_dir() and (parent / "artifacts").is_dir()),
    SCRIPT_DIR.parents[4],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/debug/028_fakevlm_linear_time_proportion"
DEFAULT_MODEL_PATH = "/home/agent/wja/data/models/lingcco/fakeVLM"
DEFAULT_TEST_JSON = "/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json"
DEFAULT_IMAGE_ROOT = "/home/agent/wja/data/datasets/lingcco/FakeClue/test/test"

WORKLOADS = {
    "prefill_b1_i1024": {"batch_size": 1, "input_tokens": 1024, "output_tokens": 0},
    "prefill_b4_i1024": {"batch_size": 4, "input_tokens": 1024, "output_tokens": 0},
    "prefill_b16_i1024": {"batch_size": 16, "input_tokens": 1024, "output_tokens": 0},
    "prefill_b4_i4096": {"batch_size": 4, "input_tokens": 4096, "output_tokens": 0},
    "normal_01": {"batch_size": 1, "input_tokens": 16384, "output_tokens": 32},
    "normal_02": {"batch_size": 1, "input_tokens": 16384, "output_tokens": 256},
}


@dataclass(frozen=True)
class CycleResult:
    prefill_ms: float
    decode_total_ms: float
    decode_avg_ms: float
    decode_first_ms: float
    decode_steady_ms: float
    e2e_ms: float


class FakeVLMDataset(Dataset):
    def __init__(self, *, model_path: str, test_json_file: str, image_root: str, sample_limit: int | None, input_tokens: int) -> None:
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


class HookTimingCollector:
    def __init__(self) -> None:
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.handles: list[Any] = []

    def register(self, module: nn.Module, label: str) -> None:
        start_holder: list[torch.cuda.Event | None] = [None]

        def pre_hook(_module: nn.Module, _inputs: Any) -> None:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            start_holder[0] = start

        def post_hook(_module: nn.Module, _inputs: Any, _outputs: Any) -> None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if start_holder[0] is not None:
                self.events[label].append((start_holder[0], end))

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def collect_all(self) -> dict[str, float]:
        torch.cuda.synchronize()
        out: dict[str, float] = {}
        for label, pairs in self.events.items():
            out[label] = sum(start.elapsed_time(end) for start, end in pairs)
        self.events.clear()
        return out

    def drain(self) -> None:
        self.events.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.events.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure FakeVLM nn.Linear time proportion.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "results").mkdir(parents=True, exist_ok=True)
    (args.output_root / "status").mkdir(parents=True, exist_ok=True)

    raw_csv = args.output_root / "results" / "fakevlm_linear_proportion_raw.csv"
    if not args.overwrite and has_workload(raw_csv, args.workload):
        print(f"[skip] existing workload={args.workload}")
        return

    workload = dict(WORKLOADS[args.workload])
    batch = load_batch(args, workload, device)
    model = load_fakevlm(args.model_path, device)
    linear_counts = count_linears(model)
    speed = benchmark_speed(model, batch, workload["output_tokens"], warmup=args.warmup, iters=args.iters)
    breakdown = benchmark_breakdown(model, batch, workload["output_tokens"], warmup=args.warmup, iters=args.iters)
    row = result_row(args, workload, batch, linear_counts, speed, breakdown)
    append_csv(raw_csv, row)
    write_json(args.output_root / "status" / f"{args.workload}.json", {"state": "done", "row": row})
    print(f"[done] workload={args.workload} prefill={speed.prefill_ms:.3f} e2e={speed.e2e_ms:.3f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_batch(args: argparse.Namespace, workload: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]:
    sample_limit = args.sample_limit
    if sample_limit is None:
        sample_limit = max(int(workload["batch_size"]), 1)
    dataset = FakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=sample_limit,
        input_tokens=int(workload["input_tokens"]),
    )
    loader = DataLoader(dataset, batch_size=int(workload["batch_size"]), shuffle=False, num_workers=args.workers, pin_memory=True)
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("FakeVLM dataset is empty") from exc
    return move_inputs(batch, device)


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


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


@torch.inference_mode()
def benchmark_speed(model: nn.Module, batch: dict[str, torch.Tensor], output_tokens: int, *, warmup: int, iters: int) -> CycleResult:
    for _ in range(warmup):
        run_one_cycle(model, batch, min(output_tokens, 2))
    torch.cuda.synchronize()
    rows = [run_one_cycle(model, batch, output_tokens) for _ in range(iters)]
    return average_cycles(rows)


def run_one_cycle(model: nn.Module, batch: dict[str, torch.Tensor], output_tokens: int) -> CycleResult:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = model(**batch, use_cache=output_tokens > 0)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = float(start.elapsed_time(end))
    logits = out.logits
    assert_finite(logits[:, -1:])
    if output_tokens <= 0:
        return CycleResult(prefill_ms, 0.0, 0.0, 0.0, 0.0, prefill_ms)

    past_key_values = out.past_key_values
    next_token = logits[:, -1:].argmax(dim=-1)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.clone()
    times: list[float] = []
    for _ in range(output_tokens):
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )
        decode_inputs: dict[str, Any] = {"input_ids": next_token, "past_key_values": past_key_values, "use_cache": True}
        if attention_mask is not None:
            decode_inputs["attention_mask"] = attention_mask
        start.record()
        out = model(**decode_inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
        logits = out.logits
        assert_finite(logits[:, -1:])
        past_key_values = out.past_key_values
        next_token = logits[:, -1:].argmax(dim=-1)
    decode_total = sum(times)
    decode_avg = decode_total / len(times)
    decode_steady = sum(times[1:]) / max(len(times) - 1, 1)
    return CycleResult(prefill_ms, decode_total, decode_avg, times[0], decode_steady, prefill_ms + decode_total)


def average_cycles(rows: list[CycleResult]) -> CycleResult:
    return CycleResult(
        prefill_ms=statistics.fmean(row.prefill_ms for row in rows),
        decode_total_ms=statistics.fmean(row.decode_total_ms for row in rows),
        decode_avg_ms=statistics.fmean(row.decode_avg_ms for row in rows),
        decode_first_ms=statistics.fmean(row.decode_first_ms for row in rows),
        decode_steady_ms=statistics.fmean(row.decode_steady_ms for row in rows),
        e2e_ms=statistics.fmean(row.e2e_ms for row in rows),
    )


@torch.inference_mode()
def benchmark_breakdown(model: nn.Module, batch: dict[str, torch.Tensor], output_tokens: int, *, warmup: int, iters: int) -> dict[str, float]:
    collector = HookTimingCollector()
    install_linear_hooks(model, collector)
    try:
        for _ in range(warmup):
            run_one_breakdown_cycle(model, batch, min(output_tokens, 1), collector)
        torch.cuda.synchronize()
        prefill_rows = []
        decode_rows = []
        for _ in range(iters):
            prefill, decode = run_one_breakdown_cycle(model, batch, min(output_tokens, 1), collector)
            prefill_rows.append(prefill)
            decode_rows.append(decode)
    finally:
        collector.close()
    prefill_avg = average_dicts(prefill_rows)
    decode_avg = average_dicts(decode_rows)
    return breakdown_metrics(prefill_avg, decode_avg)


def run_one_breakdown_cycle(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    output_tokens: int,
    collector: HookTimingCollector,
) -> tuple[dict[str, float], dict[str, float]]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = model(**batch, use_cache=output_tokens > 0)
    end.record()
    prefill = collector.collect_all()
    prefill["_total_ms"] = float(start.elapsed_time(end))
    logits = out.logits
    assert_finite(logits[:, -1:])
    if output_tokens <= 0:
        collector.drain()
        return prefill, {"_total_ms": 0.0}

    past_key_values = out.past_key_values
    next_token = logits[:, -1:].argmax(dim=-1)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.clone()
    decode: dict[str, float] = {}
    for step in range(output_tokens):
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )
        decode_inputs: dict[str, Any] = {"input_ids": next_token, "past_key_values": past_key_values, "use_cache": True}
        if attention_mask is not None:
            decode_inputs["attention_mask"] = attention_mask
        if step == 0:
            start.record()
        out = model(**decode_inputs)
        if step == 0:
            end.record()
            decode = collector.collect_all()
            decode["_total_ms"] = float(start.elapsed_time(end))
        logits = out.logits
        assert_finite(logits[:, -1:])
        past_key_values = out.past_key_values
        next_token = logits[:, -1:].argmax(dim=-1)
    collector.drain()
    if not decode:
        decode["_total_ms"] = 0.0
    return prefill, decode


def install_linear_hooks(model: nn.Module, collector: HookTimingCollector) -> None:
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        collector.register(module, "all_linear")
        collector.register(module, classify_linear_name(name))


def classify_linear_name(name: str) -> str:
    if name.startswith("vision_tower") or ".vision_tower." in name:
        return "vision_linear"
    if name.startswith("multi_modal_projector") or ".multi_modal_projector." in name:
        return "projector_linear"
    if name.startswith("language_model") or ".language_model." in name:
        return "language_linear"
    return "other_linear"


def count_linears(model: nn.Module) -> dict[str, int]:
    counts = {"all_linear_count": 0, "language_linear_count": 0, "vision_linear_count": 0, "projector_linear_count": 0, "other_linear_count": 0}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        counts["all_linear_count"] += 1
        counts[f"{classify_linear_name(name)}_count"] += 1
    return counts


def breakdown_metrics(prefill: dict[str, float], decode: dict[str, float]) -> dict[str, float]:
    out = {"prefill_total_ms": prefill.get("_total_ms", 0.0), "decode_total_hook_ms": decode.get("_total_ms", 0.0)}
    for phase, raw in (("prefill", prefill), ("decode", decode)):
        total = raw.get("_total_ms", 0.0)
        for label in ("all_linear", "language_linear", "vision_linear", "projector_linear", "other_linear"):
            out[f"{phase}_{label}_ms"] = raw.get(label, 0.0)
            out[f"{phase}_{label}_pct"] = pct(raw.get(label, 0.0), total)
        out[f"{phase}_non_linear_pct"] = max(0.0, 100.0 - out[f"{phase}_all_linear_pct"])
    return out


def result_row(
    args: argparse.Namespace,
    workload: dict[str, int],
    batch: dict[str, torch.Tensor],
    linear_counts: dict[str, int],
    speed: CycleResult,
    breakdown: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "FakeVLM",
        "workload": args.workload,
        "batch_size": workload["batch_size"],
        "actual_batch_size": int(batch["input_ids"].shape[0]),
        "input_tokens": workload["input_tokens"],
        "actual_input_tokens": int(batch["input_ids"].shape[1]),
        "output_tokens": workload["output_tokens"],
        "warmup": args.warmup,
        "iters": args.iters,
        "prefill_ms": speed.prefill_ms,
        "decode_total_ms": speed.decode_total_ms,
        "decode_avg_ms": speed.decode_avg_ms,
        "decode_first_ms": speed.decode_first_ms,
        "decode_steady_ms": speed.decode_steady_ms,
        "e2e_ms": speed.e2e_ms,
        "samples_per_sec": int(batch["input_ids"].shape[0]) * 1000.0 / speed.e2e_ms if speed.e2e_ms > 0 else 0.0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    row.update(linear_counts)
    row.update(breakdown)
    return row


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row)
    return {key: statistics.fmean(row.get(key, 0.0) for row in rows) for key in keys}


def pct(value: float, total: float) -> float:
    return 0.0 if total <= 0 else value * 100.0 / total


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and path.stat().st_size > 0:
            rows = []
            with path.open(newline="", encoding="utf-8") as f:
                for old in csv.DictReader(f):
                    if old.get("workload") != row["workload"]:
                        rows.append(old)
            rows.append({key: serialize(value) for key, value in row.items()})
            fields = list(row)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow({key: serialize(value) for key, value in row.items()})
        fcntl.flock(lock, fcntl.LOCK_UN)


def has_workload(path: Path, workload: str) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        return any(row.get("workload") == workload for row in csv.DictReader(f))


def serialize(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def assert_finite(tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all().item():
        raise RuntimeError("non-finite tensor encountered")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
