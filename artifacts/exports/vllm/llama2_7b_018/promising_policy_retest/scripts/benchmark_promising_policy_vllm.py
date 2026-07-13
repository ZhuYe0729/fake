#!/usr/bin/env python3
"""Benchmark one or more promising scenarios with optimized hetero policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


SCRIPT_DIR = Path(__file__).resolve().parent
RETEST_ROOT = SCRIPT_DIR.parent
BASELINE_ROOT = RETEST_ROOT.parent
DEFAULT_POLICY_CSV = RETEST_ROOT / "policies/scenario_policies.csv"
DEFAULT_CHECKPOINT_ROOT = RETEST_ROOT / "checkpoints"


@dataclass(frozen=True)
class Scenario:
    name: str
    batch: int
    input_seq: int
    output_seq: int
    policy_name: str

    @property
    def max_model_len(self) -> int:
        return self.input_seq + self.output_seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-csv", type=Path, default=DEFAULT_POLICY_CSV)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="optimized_hetero")
    parser.add_argument("--output-prefix", default="optimized_hetero")
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = filter_scenarios(read_scenarios(args.policy_csv), args.scenarios)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    for scenario in scenarios:
        try:
            summary, iter_rows = benchmark_scenario(args, scenario)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            summary = base_summary(scenario, args.checkpoint_root / scenario.policy_name, args.method_name)
            summary.update({"status": classify_exception(exc), "error": " ".join(str(exc).split())[:500]})
            iter_rows = []
            print(f"ERROR {scenario.name}: {type(exc).__name__}: {exc}", flush=True)
        summaries.append(summary)
        iterations.extend(iter_rows)
        write_csv(args.output_dir / f"{args.output_prefix}_summary.csv", summaries)
        write_csv(args.output_dir / f"{args.output_prefix}_iterations.csv", iterations)
    write_json(
        args.output_dir / f"{args.output_prefix}_metadata.json",
        {
            "scenarios": [scenario.__dict__ for scenario in scenarios],
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": not args.no_enforce_eager,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )


def read_scenarios(path: Path) -> list[Scenario]:
    out: list[Scenario] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out.append(
                Scenario(
                    name=row["scenario"],
                    batch=int(row["batch"]),
                    input_seq=int(row["input_seq"]),
                    output_seq=int(row["output_seq"]),
                    policy_name=row["policy_name"],
                )
            )
    return out


def filter_scenarios(scenarios: list[Scenario], spec: str) -> list[Scenario]:
    if spec == "all":
        return scenarios
    names = [item.strip() for item in spec.split(",") if item.strip()]
    by_name = {scenario.name: scenario for scenario in scenarios}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown scenarios: {missing}")
    return [by_name[name] for name in names]


def benchmark_scenario(args: argparse.Namespace, scenario: Scenario) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_path = args.checkpoint_root / scenario.policy_name
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    print(
        f"loading {args.method_name} {scenario.name} policy={scenario.policy_name} "
        f"max_model_len={scenario.max_model_len} batch={scenario.batch}",
        flush=True,
    )
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=scenario.max_model_len,
        max_num_seqs=scenario.batch,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        enable_prefix_caching=False,
        hf_overrides=long_context_hf_overrides(model_path, scenario.max_model_len),
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000))
    prompts = make_prompts(scenario.batch, scenario.input_seq, vocab_size, args.seed)
    sampling = SamplingParams(
        max_tokens=scenario.output_seq,
        min_tokens=scenario.output_seq,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    for i in range(args.warmup_iters):
        run_once(llm, prompts, sampling)
        print(f"{args.method_name} {scenario.name} warmup {i + 1}/{args.warmup_iters}", flush=True)
    iter_rows: list[dict[str, Any]] = []
    for i in range(args.iters):
        elapsed = run_once(llm, prompts, sampling)
        row = {
            "method": args.method_name,
            "scenario": scenario.name,
            "iter": i,
            "elapsed_ms": elapsed * 1000.0,
            "batch": scenario.batch,
            "input_seq": scenario.input_seq,
            "output_seq": scenario.output_seq,
            "policy_name": scenario.policy_name,
            "model_path": str(model_path),
        }
        iter_rows.append(row)
        print(f"{args.method_name} {scenario.name} iter {i + 1}/{args.iters}: {row['elapsed_ms']:.3f} ms", flush=True)
    summary = summarize(scenario, model_path, iter_rows, args.method_name)
    del llm
    torch.cuda.empty_cache()
    return summary, iter_rows


def long_context_hf_overrides(model_path: Path, max_model_len: int) -> dict[str, Any] | None:
    try:
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
        current = int(getattr(config, "max_position_embeddings", 0) or 0)
    except Exception:
        current = 0
    if max_model_len > current:
        return {"max_position_embeddings": max_model_len}
    return None


def make_prompts(batch: int, prompt_len: int, vocab_size: int, seed: int) -> list[TokensPrompt]:
    usable = max(100, min(vocab_size - 1, 30000))
    prompts = []
    for row in range(batch):
        ids = [1 + ((seed * 1000003 + row * 9176 + pos) % usable) for pos in range(prompt_len)]
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    return prompts


def run_once(llm: LLM, prompts: list[TokensPrompt], sampling: SamplingParams) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = llm.generate(prompts, sampling, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def summarize(scenario: Scenario, model_path: Path, rows: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
    values = [float(row["elapsed_ms"]) for row in rows]
    summary = base_summary(scenario, model_path, method_name)
    summary.update(
        {
            "status": "OK",
            "iters": len(values),
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "error": "",
        }
    )
    return summary


def base_summary(scenario: Scenario, model_path: Path, method_name: str) -> dict[str, Any]:
    return {
        "method": method_name,
        "scenario": scenario.name,
        "batch": scenario.batch,
        "input_seq": scenario.input_seq,
        "output_seq": scenario.output_seq,
        "policy_name": scenario.policy_name,
        "model_path": str(model_path),
    }


def classify_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "OOM"
    if "kv cache" in text or "maximum model length" in text:
        return "INIT_ERROR"
    return "ERROR"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
