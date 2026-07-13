#!/usr/bin/env python3
"""Run one-method Llama2 broad-grid vLLM latency benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


SCRIPT_DIR = Path(__file__).resolve().parent
BROAD_ROOT = SCRIPT_DIR.parent
BASELINE_ROOT = BROAD_ROOT.parent
REPO_ROOT = BASELINE_ROOT.parents[3]
DEFAULT_DENSE_BF16_MODEL = Path("/root/wja/data/models/LLM-Research/llama-2-7b")

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
INPUT_SEQS = (128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536)
OUTPUT_SEQS = (1, 16, 64, 128)

MODEL_SPECS = {
    "dense_bf16": DEFAULT_DENSE_BF16_MODEL,
    "dense_nvfp4": BASELINE_ROOT / "uniform_dense_nvfp4",
    "sparse_bf16": BASELINE_ROOT / "uniform_sparse_bf16",
    "sparse_nvfp4": BASELINE_ROOT / "uniform_sparse_nvfp4",
    "marlin_nvfp4": BASELINE_ROOT / "uniform_marlin_nvfp4",
    "hetero_strategy_a": BASELINE_ROOT / "hetero_strategy_a",
    "hetero_strategy_b": BASELINE_ROOT / "hetero_strategy_b",
    "hetero_strategy_c": BASELINE_ROOT / "hetero_strategy_c",
}
METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "hetero",
)


@dataclass(frozen=True)
class GridConfig:
    batch: int
    input_seq: int
    output_seq: int

    @property
    def name(self) -> str:
        return f"b{self.batch}_in{self.input_seq}_out{self.output_seq}"

    @property
    def max_model_len(self) -> int:
        return self.input_seq + self.output_seq

    @property
    def total_prompt_tokens(self) -> int:
        return self.batch * self.input_seq

    @property
    def total_tokens(self) -> int:
        return self.batch * (self.input_seq + self.output_seq)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    parser.add_argument("--input-seqs", default=",".join(map(str, INPUT_SEQS)))
    parser.add_argument("--output-seqs", default=",".join(map(str, OUTPUT_SEQS)))
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-total-prompt-tokens", type=int, default=131072)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--no-hf-max-position-override", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.pop("_CUDA_COMPAT_STATUS", None)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = make_grid(args)
    rows: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []

    for group_key, group_configs in group_configs_for_method(args.method, configs).items():
        group_rows, group_iterations = benchmark_group(group_key, group_configs, args)
        rows.extend(group_rows)
        iterations.extend(group_iterations)
        write_csv(args.output_dir / "summary_long.csv", rows)
        write_csv(args.output_dir / "iterations.csv", iterations)

    write_json(
        args.output_dir / "metadata.json",
        {
            "method": args.method,
            "batches": parse_ints(args.batches),
            "input_seqs": parse_ints(args.input_seqs),
            "output_seqs": parse_ints(args.output_seqs),
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": not args.no_enforce_eager,
            "max_total_prompt_tokens": args.max_total_prompt_tokens,
            "device": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )


def parse_ints(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def make_grid(args: argparse.Namespace) -> list[GridConfig]:
    return [
        GridConfig(batch=batch, input_seq=input_seq, output_seq=output_seq)
        for input_seq in parse_ints(args.input_seqs)
        for output_seq in parse_ints(args.output_seqs)
        for batch in parse_ints(args.batches)
    ]


def group_configs_for_method(
    method: str, configs: list[GridConfig]
) -> dict[tuple[str, int, int], list[GridConfig]]:
    groups: dict[tuple[str, int, int], list[GridConfig]] = {}
    for config in configs:
        model_key = model_key_for_config(method, config)
        groups.setdefault((model_key, config.input_seq, config.output_seq), []).append(config)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda item: item.batch)
    return groups


def model_key_for_config(method: str, config: GridConfig) -> str:
    if method != "hetero":
        return method
    prefill_m = config.batch * config.input_seq
    if config.batch <= 2 and prefill_m <= 8192:
        return "hetero_strategy_a"
    if config.batch <= 4 and prefill_m <= 16384:
        return "hetero_strategy_b"
    return "hetero_strategy_c"


def benchmark_group(
    group_key: tuple[str, int, int],
    configs: list[GridConfig],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_key, input_seq, output_seq = group_key
    max_model_len = input_seq + output_seq
    model_path = MODEL_SPECS[model_key]
    rows: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    runnable = [c for c in configs if c.total_prompt_tokens <= args.max_total_prompt_tokens]
    skipped = [c for c in configs if c.total_prompt_tokens > args.max_total_prompt_tokens]
    for config in skipped:
        rows.append(status_row(args.method, model_key, model_path, config, "PRECHECK_OOM"))
    if not runnable:
        return rows, iterations

    llm = None
    try:
        print(
            f"loading method={args.method} model={model_key} "
            f"input={input_seq} output={output_seq} max_model_len={max_model_len} "
            f"max_batch={max(c.batch for c in runnable)}",
            flush=True,
        )
        llm = load_llm(model_path, max_model_len, max(c.batch for c in runnable), args)
        tokenizer = llm.get_tokenizer()
        vocab_size = int(getattr(tokenizer, "vocab_size", 32000))
    except Exception as exc:
        status = classify_exception(exc, init=True)
        error = format_exception(exc)
        print(f"INIT_FAIL {args.method} {group_key}: {status}: {exc}", flush=True)
        for config in runnable:
            rows.append(status_row(args.method, model_key, model_path, config, status, error))
        cleanup_llm(llm)
        return rows, iterations

    stop_after_oom = False
    for config in runnable:
        if stop_after_oom:
            rows.append(status_row(args.method, model_key, model_path, config, "OOM_SKIPPED"))
            continue
        try:
            summary, iter_rows = benchmark_config(
                llm=llm,
                method=args.method,
                model_key=model_key,
                model_path=model_path,
                config=config,
                vocab_size=vocab_size,
                args=args,
            )
            rows.append(summary)
            iterations.extend(iter_rows)
        except Exception as exc:
            status = classify_exception(exc, init=False)
            error = format_exception(exc)
            print(f"RUN_FAIL {args.method} {config.name}: {status}: {exc}", flush=True)
            rows.append(status_row(args.method, model_key, model_path, config, status, error))
            cleanup_cuda()
            if status == "OOM":
                stop_after_oom = True

    cleanup_llm(llm)
    return rows, iterations


def load_llm(
    model_path: Path,
    max_model_len: int,
    max_num_seqs: int,
    args: argparse.Namespace,
) -> LLM:
    hf_overrides = long_context_hf_overrides(model_path, max_model_len, args)
    return LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        enable_prefix_caching=False,
        hf_overrides=hf_overrides,
    )


def long_context_hf_overrides(
    model_path: Path, max_model_len: int, args: argparse.Namespace
) -> dict[str, Any] | None:
    if args.no_hf_max_position_override:
        return None
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    configured = int(getattr(config, "max_position_embeddings", 0) or 0)
    if configured and max_model_len > configured:
        return {"max_position_embeddings": max_model_len}
    return None


def benchmark_config(
    *,
    llm: LLM,
    method: str,
    model_key: str,
    model_path: Path,
    config: GridConfig,
    vocab_size: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = make_prompts(
        batch_size=config.batch,
        prompt_len=config.input_seq,
        vocab_size=vocab_size,
        seed=args.seed,
    )
    sampling = SamplingParams(
        max_tokens=config.output_seq,
        min_tokens=config.output_seq,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    for i in range(args.warmup_iters):
        run_once(llm, prompts, sampling)
        print(f"{method} {config.name} warmup {i + 1}/{args.warmup_iters}", flush=True)
    iter_rows: list[dict[str, Any]] = []
    for i in range(args.iters):
        elapsed = run_once(llm, prompts, sampling)
        row = base_row(method, model_key, model_path, config)
        row.update({"iter": i, "elapsed_ms": elapsed * 1000.0})
        iter_rows.append(row)
        print(f"{method} {config.name} iter {i + 1}/{args.iters}: {row['elapsed_ms']:.3f} ms", flush=True)

    times = [float(row["elapsed_ms"]) for row in iter_rows]
    summary = base_row(method, model_key, model_path, config)
    median_ms = statistics.median(times)
    mean_ms = statistics.mean(times)
    summary.update(
        {
            "status": "OK",
            "iters": len(iter_rows),
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "input_tokens_per_s_median": config.total_prompt_tokens / (median_ms / 1000.0),
            "total_tokens_per_s_median": config.total_tokens / (median_ms / 1000.0),
            "error": "",
        }
    )
    return summary, iter_rows


def make_prompts(
    *, batch_size: int, prompt_len: int, vocab_size: int, seed: int
) -> list[TokensPrompt]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    low = 100
    high = max(low + 1, vocab_size - 1)
    token_blocks = torch.randint(
        low=low,
        high=high,
        size=(batch_size, prompt_len),
        generator=generator,
        dtype=torch.int32,
    )
    return [TokensPrompt(prompt_token_ids=row.tolist()) for row in token_blocks]


def run_once(llm: LLM, prompts: list[TokensPrompt], sampling: SamplingParams) -> float:
    start = time.perf_counter()
    llm.generate(prompts, sampling, use_tqdm=False)
    return time.perf_counter() - start


def base_row(method: str, model_key: str, model_path: Path, config: GridConfig) -> dict[str, Any]:
    return {
        "method": method,
        "model_key": model_key,
        "model_path": str(model_path),
        "batch": config.batch,
        "input_seq": config.input_seq,
        "output_seq": config.output_seq,
        "max_model_len": config.max_model_len,
        "total_prompt_tokens": config.total_prompt_tokens,
        "total_tokens": config.total_tokens,
    }


def status_row(
    method: str,
    model_key: str,
    model_path: Path,
    config: GridConfig,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    row = base_row(method, model_key, model_path, config)
    row.update(
        {
            "status": status,
            "iters": 0,
            "mean_ms": math.nan,
            "median_ms": math.nan,
            "min_ms": math.nan,
            "max_ms": math.nan,
            "std_ms": math.nan,
            "input_tokens_per_s_median": math.nan,
            "total_tokens_per_s_median": math.nan,
            "error": error,
        }
    )
    return row


def classify_exception(exc: Exception, *, init: bool) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "oom" in text or "kv cache" in text or "no available memory" in text:
        return "INIT_OOM" if init else "OOM"
    if "maximum model length" in text or "max seq len" in text:
        return "INIT_OOM" if init else "OOM"
    return "INIT_ERROR" if init else "ERROR"


def format_exception(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def cleanup_llm(llm: Any) -> None:
    if llm is not None:
        del llm
    cleanup_cuda()


def cleanup_cuda() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
