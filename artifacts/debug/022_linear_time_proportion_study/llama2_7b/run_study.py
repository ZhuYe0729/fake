#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_DIR = REPO_ROOT / "artifacts/debug/022_linear_time_proportion_study/llama2_7b"
MODEL_CANDIDATES = (
    Path(os.environ.get("LLAMA2_MODEL_PATH", "")) if os.environ.get("LLAMA2_MODEL_PATH") else None,
    Path("/data/home/scxj523/run/wja/data/models/LLM-Research/llama-2-7b"),
    Path("/home/agent/wja/data/models/LLM-Research/llama-2-7b"),
)

BATCH_SIZES = (1, 4, 16, 32, 64)
INPUT_TOKENS = (16, 64, 256, 1024, 4096, 8192)
OUTPUT_TOKENS_SPEED = (1, 32, 128, 256)
OUTPUT_TOKENS_BREAKDOWN = (1, 32)


class HookTimingCollector:
    def __init__(self) -> None:
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.handles: list[Any] = []

    def register(self, module: nn.Module, label: str) -> None:
        start_holder: list[torch.cuda.Event | None] = [None]

        def pre_hook(_mod: nn.Module, _inp: Any) -> None:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            start_holder[0] = start

        def post_hook(_mod: nn.Module, _inp: Any, _out: Any) -> None:
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
    parser = argparse.ArgumentParser(description="Llama2-7B dense BF16 linear time proportion study.")
    parser.add_argument("--phase", choices=("speed", "breakdown"), required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--input-mode", choices=("random",), default="random")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Llama2-7B timing")

    torch.cuda.set_device(args.gpu)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("speed", "breakdown_coarse", "logs", "summary"):
        (ARTIFACT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model_path)
    print(f"Phase: {args.phase}")
    print(f"Model path: {model_path}")
    print(f"GPU index inside process: {args.gpu}")
    print(f"Device: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Shard: {args.shard_index}/{args.num_shards}")
    print(f"Warmup={args.warmup}, iters={args.iters}")
    print()

    model = load_model(model_path, args.gpu, args.attn_implementation)
    print_model_summary(model)
    configs = make_configs(args.phase)
    shard_configs = [cfg for i, cfg in enumerate(configs) if i % args.num_shards == args.shard_index]
    print(f"Total configs={len(configs)}, shard configs={len(shard_configs)}")

    if args.phase == "speed":
        out_csv = ARTIFACT_DIR / "speed" / f"llama2_7b_speed_shard{args.shard_index}.csv"
        run_speed_phase(model, shard_configs, args, out_csv)
    else:
        out_csv = ARTIFACT_DIR / "breakdown_coarse" / f"llama2_7b_breakdown_coarse_shard{args.shard_index}.csv"
        run_breakdown_phase(model, shard_configs, args, out_csv)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


def resolve_model_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in MODEL_CANDIDATES:
        if candidate is not None and str(candidate) and candidate.exists():
            return candidate
    fallback = Path("/home/agent/wja/data/models/LLM-Research/llama-2-7b")
    print(f"WARNING: no default model path exists; using fallback {fallback}")
    return fallback


def load_model(model_path: Path, gpu: int, attn_implementation: str) -> nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    model = model.to(f"cuda:{gpu}")
    model.eval()
    model.requires_grad_(False)
    return model


def print_model_summary(model: nn.Module) -> None:
    linear_count = sum(1 for module in model.modules() if isinstance(module, nn.Linear))
    layer_count = len(get_decoder_layers(model))
    vocab_size = int(getattr(model.config, "vocab_size", 32000))
    print(f"Model: {type(model).__name__}")
    print(f"Decoder layers: {layer_count}")
    print(f"nn.Linear modules: {linear_count}")
    print(f"Vocab size: {vocab_size}")
    print(f"Dtype: bf16")
    print()


def make_configs(phase: str) -> list[tuple[int, int, int]]:
    output_tokens = OUTPUT_TOKENS_SPEED if phase == "speed" else OUTPUT_TOKENS_BREAKDOWN
    return [(bs, itok, otok) for bs in BATCH_SIZES for itok in INPUT_TOKENS for otok in output_tokens]


def get_vocab_size(model: nn.Module) -> int:
    return int(getattr(model.config, "vocab_size", 32000))


def make_input_ids(model: nn.Module, batch_size: int, input_tokens: int, device: str) -> torch.Tensor:
    high = min(get_vocab_size(model), 32000)
    return torch.randint(0, high, (batch_size, input_tokens), device=device)


def run_speed_phase(
    model: nn.Module,
    configs: list[tuple[int, int, int]],
    args: argparse.Namespace,
    out_csv: Path,
) -> None:
    fieldnames = [
        "model",
        "batch_size",
        "input_tokens",
        "output_tokens",
        "prefill_ms",
        "decode_total_ms",
        "decode_per_token_ms",
        "first_decode_ms",
        "tokens_per_sec",
        "warmup",
        "iters",
        "gpu",
        "status",
        "error_msg",
    ]
    writer, file_obj = open_csv(out_csv, fieldnames)
    try:
        for idx, (batch_size, input_tokens, output_tokens) in enumerate(configs, 1):
            print(f"[speed {idx}/{len(configs)}] bs={batch_size} input={input_tokens} output={output_tokens}", flush=True)
            row = base_row(batch_size, input_tokens, output_tokens, args)
            try:
                metrics = benchmark_speed(model, batch_size, input_tokens, output_tokens, args)
                row.update(metrics)
                row["status"] = "OK"
                print(
                    f"  prefill={metrics['prefill_ms']:.3f}ms "
                    f"decode/tok={metrics['decode_per_token_ms']:.3f}ms",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as exc:
                row.update(error_fields("OOM", exc))
                torch.cuda.empty_cache()
                print("  OOM", flush=True)
            except Exception as exc:  # keep the sweep moving
                row.update(error_fields("ERROR", exc))
                torch.cuda.empty_cache()
                print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
            writer.writerow(row)
            file_obj.flush()
    finally:
        file_obj.close()


@torch.inference_mode()
def benchmark_speed(
    model: nn.Module,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    device = f"cuda:{args.gpu}"
    for _ in range(args.warmup):
        _do_one_cycle(model, batch_size, min(input_tokens, 64), output_tokens, device)
    torch.cuda.synchronize()

    prefill_values: list[float] = []
    first_decode_values: list[float] = []
    steady_decode_values: list[float] = []
    total_decode_values: list[float] = []
    for _ in range(args.iters):
        prefill_ms, decode_ms = _do_one_cycle(model, batch_size, input_tokens, output_tokens, device)
        prefill_values.append(prefill_ms)
        total_decode_values.append(sum(decode_ms))
        if decode_ms:
            first_decode_values.append(decode_ms[0])
            steady_decode_values.extend(decode_ms[1:] if len(decode_ms) > 1 else [])

    decode_per_token = mean(steady_decode_values) if steady_decode_values else 0.0
    tokens_per_sec = batch_size * 1000.0 / decode_per_token if decode_per_token > 0 else 0.0
    return {
        "prefill_ms": mean(prefill_values),
        "decode_total_ms": mean(total_decode_values),
        "decode_per_token_ms": decode_per_token,
        "first_decode_ms": mean(first_decode_values) if first_decode_values else 0.0,
        "tokens_per_sec": tokens_per_sec,
    }


def _do_one_cycle(
    model: nn.Module,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    device: str,
) -> tuple[float, list[float]]:
    input_ids = make_input_ids(model, batch_size, input_tokens, device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    outputs = model(input_ids=input_ids, use_cache=True)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = float(start.elapsed_time(end))

    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1:, :]
    decode_ms: list[float] = []
    for _ in range(output_tokens):
        next_token = next_logits.argmax(dim=-1)
        start.record()
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        end.record()
        torch.cuda.synchronize()
        decode_ms.append(float(start.elapsed_time(end)))
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1:, :]
    return prefill_ms, decode_ms


def run_breakdown_phase(
    model: nn.Module,
    configs: list[tuple[int, int, int]],
    args: argparse.Namespace,
    out_csv: Path,
) -> None:
    fieldnames = [
        "model",
        "batch_size",
        "input_tokens",
        "output_tokens",
        "prefill_total_ms",
        "decode_total_ms",
        "prefill_all_linear_pct",
        "decode_all_linear_pct",
        "prefill_self_attn_block_pct",
        "decode_self_attn_block_pct",
        "prefill_mlp_block_pct",
        "decode_mlp_block_pct",
        "prefill_norm_pct",
        "decode_norm_pct",
        "prefill_lm_head_pct",
        "decode_lm_head_pct",
        "prefill_other_pct",
        "decode_other_pct",
        "warmup",
        "iters",
        "gpu",
        "status",
        "error_msg",
    ]
    writer, file_obj = open_csv(out_csv, fieldnames)
    collector = HookTimingCollector()
    install_coarse_hooks(model, collector)
    try:
        for idx, (batch_size, input_tokens, output_tokens) in enumerate(configs, 1):
            print(f"[breakdown {idx}/{len(configs)}] bs={batch_size} input={input_tokens} output={output_tokens}", flush=True)
            row = base_row(batch_size, input_tokens, output_tokens, args)
            try:
                metrics = benchmark_breakdown(model, collector, batch_size, input_tokens, output_tokens, args)
                row.update(metrics)
                row["status"] = "OK"
                print(
                    f"  prefill_total={metrics['prefill_total_ms']:.3f}ms "
                    f"linear={metrics['prefill_all_linear_pct']:.1f}%",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as exc:
                collector.drain()
                row.update(error_fields("OOM", exc))
                torch.cuda.empty_cache()
                print("  OOM", flush=True)
            except Exception as exc:
                collector.drain()
                row.update(error_fields("ERROR", exc))
                torch.cuda.empty_cache()
                print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
            writer.writerow(row)
            file_obj.flush()
    finally:
        collector.close()
        file_obj.close()


def install_coarse_hooks(model: nn.Module, collector: HookTimingCollector) -> None:
    for layer in get_decoder_layers(model):
        if hasattr(layer, "self_attn"):
            collector.register(layer.self_attn, "self_attn_block")
        if hasattr(layer, "mlp"):
            collector.register(layer.mlp, "mlp_block")
        if hasattr(layer, "input_layernorm"):
            collector.register(layer.input_layernorm, "norm")
        if hasattr(layer, "post_attention_layernorm"):
            collector.register(layer.post_attention_layernorm, "norm")
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "norm"):
        collector.register(inner.norm, "norm")
    if hasattr(model, "lm_head") and model.lm_head is not None:
        collector.register(model.lm_head, "lm_head")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            collector.register(module, "_all_linear_internal")


def get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise RuntimeError("Cannot find decoder layers")


@torch.inference_mode()
def benchmark_breakdown(
    model: nn.Module,
    collector: HookTimingCollector,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    device = f"cuda:{args.gpu}"
    for _ in range(args.warmup):
        _do_one_breakdown_cycle(model, collector, batch_size, min(input_tokens, 64), output_tokens, device)
    torch.cuda.synchronize()

    prefill_results: list[dict[str, float]] = []
    decode_results: list[dict[str, float]] = []
    for _ in range(args.iters):
        prefill_raw, decode_raw = _do_one_breakdown_cycle(
            model, collector, batch_size, input_tokens, output_tokens, device
        )
        prefill_results.append(prefill_raw)
        decode_results.append(decode_raw)

    prefill_avg = average_dicts(prefill_results)
    decode_avg = average_dicts(decode_results)
    prefill_avg["all_linear"] = prefill_avg.pop("_all_linear_internal", 0.0)
    decode_avg["all_linear"] = decode_avg.pop("_all_linear_internal", 0.0)
    prefill_total = prefill_avg.pop("_total_ms", 0.0)
    decode_total = decode_avg.pop("_total_ms", 0.0)
    prefill_pct = to_pct(prefill_avg, prefill_total)
    decode_pct = to_pct(decode_avg, decode_total)
    add_other(prefill_pct)
    add_other(decode_pct)

    out: dict[str, float] = {
        "prefill_total_ms": prefill_total,
        "decode_total_ms": decode_total,
    }
    for label in ("all_linear", "self_attn_block", "mlp_block", "norm", "lm_head", "other"):
        out[f"prefill_{label}_pct"] = prefill_pct.get(label, 0.0)
        out[f"decode_{label}_pct"] = decode_pct.get(label, 0.0)
    return out


def _do_one_breakdown_cycle(
    model: nn.Module,
    collector: HookTimingCollector,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    device: str,
) -> tuple[dict[str, float], dict[str, float]]:
    input_ids = make_input_ids(model, batch_size, input_tokens, device)
    p_start = torch.cuda.Event(enable_timing=True)
    p_end = torch.cuda.Event(enable_timing=True)
    p_start.record()
    outputs = model(input_ids=input_ids, use_cache=True)
    p_end.record()
    prefill_raw = collector.collect_all()
    prefill_raw["_total_ms"] = float(p_start.elapsed_time(p_end))

    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1:, :]
    decode_raw: dict[str, float] = {}
    d_start = torch.cuda.Event(enable_timing=True)
    d_end = torch.cuda.Event(enable_timing=True)
    for step in range(output_tokens):
        next_token = next_logits.argmax(dim=-1)
        if step == 0:
            d_start.record()
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        if step == 0:
            d_end.record()
            decode_raw = collector.collect_all()
            decode_raw["_total_ms"] = float(d_start.elapsed_time(d_end))
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1:, :]
    if not decode_raw:
        decode_raw["_total_ms"] = 0.0
    collector.drain()
    return prefill_raw, decode_raw


def add_other(pct: dict[str, float]) -> None:
    tracked = ("self_attn_block", "mlp_block", "norm", "lm_head")
    pct["other"] = max(0.0, 100.0 - sum(pct.get(k, 0.0) for k in tracked))


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row)
    return {key: mean([row.get(key, 0.0) for row in rows]) for key in keys}


def to_pct(raw: dict[str, float], total: float) -> dict[str, float]:
    if total <= 0:
        return {key: 0.0 for key in raw}
    return {key: value * 100.0 / total for key, value in raw.items()}


def base_row(batch_size: int, input_tokens: int, output_tokens: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": "llama2-7b",
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "warmup": args.warmup,
        "iters": args.iters,
        "gpu": args.gpu,
        "status": "",
        "error_msg": "",
    }


def error_fields(status: str, exc: Exception) -> dict[str, str]:
    return {"status": status, "error_msg": f"{type(exc).__name__}: {exc}"}


def open_csv(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_obj = path.open("w", newline="")
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    file_obj.flush()
    return writer, file_obj


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
