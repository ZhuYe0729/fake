#!/usr/bin/env python3
"""Measure teacher-forced decode NLL through actual vLLM phase execution."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
VLLM = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--phase-hetero", action="store_true")
    return parser.parse_args()


def sparse_bf16_policy(model: Path) -> bool:
    policy_path = model / "phase_hetero_policy.json"
    if not policy_path.exists():
        return model.name == "uniform_sparse_bf16"
    policy = json.loads(policy_path.read_text())
    methods = [policy["default_prefill_method"], policy["default_decode_method"]]
    methods.extend(method for pair in policy["method_map"].values()
                   for method in pair.values())
    return "sparse_bf16" in methods


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(VLLM / "vllm"), str(VLLM), str(CUTLASS),
                    str(Path(__file__).parent)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    if args.phase_hetero:
        os.environ["PHASE_HETERO_TRACE"] = "1"
    sparse = sparse_bf16_policy(args.model)
    if sparse:
        os.environ["CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES"] = "16"

    from teacher_force_processor import TeacherForceBatchProcessor
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    phase = None
    if args.phase_hetero:
        from vllm.model_executor.layers.quantization import phase_hetero_mytest as phase

    samples = torch.load(args.samples, map_location="cpu")[:args.blocks]
    required = args.input_tokens + args.output_tokens
    if samples.ndim != 2 or samples.shape[1] < required:
        raise ValueError(f"samples must have >= {required} tokens per block, got {tuple(samples.shape)}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    capture_root = args.output.parent / f"{args.label}_captures"
    if capture_root.exists():
        raise FileExistsError(capture_root)
    llm = None
    started = time.perf_counter()
    rows = []
    try:
        llm = LLM(model=str(args.model), tokenizer=str(args.tokenizer), dtype="bfloat16",
                  enforce_eager=True, enable_prefix_caching=False,
                  enable_chunked_prefill=False, skip_mm_profiling=True,
                  max_model_len=required + 8, max_num_seqs=args.batch_size,
                  max_num_batched_tokens=args.batch_size * required,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  logits_processors=[TeacherForceBatchProcessor])
        if phase is not None:
            phase.enable_phase_hetero()
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            if phase is not None and start:
                phase.wait_for_prefill_ready()
            prompts, params, targets, captures = [], [], [], []
            for offset, row in enumerate(batch):
                index = start + offset
                prompt_ids = row[:args.input_tokens].tolist()
                target_ids = row[args.input_tokens:required].tolist()
                capture = capture_root / f"block_{index:04d}.jsonl"
                prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
                params.append(SamplingParams(temperature=0.0, max_tokens=args.output_tokens,
                                              extra_args={"teacher_force_target_ids": target_ids,
                                                          "teacher_force_capture_path": str(capture)}))
                targets.append(target_ids); captures.append(capture)
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for offset, (output, target_ids, capture) in enumerate(zip(outputs, targets, captures)):
                index = start + offset
                generated = list(output.outputs[0].token_ids)
                if generated != target_ids:
                    raise AssertionError(f"{args.label}/block {index}: forced decode mismatch")
                logprobs = [json.loads(line)["logprob"] for line in capture.read_text().splitlines()]
                if len(logprobs) != args.output_tokens:
                    raise AssertionError(f"{args.label}/block {index}: captured {len(logprobs)} tokens")
                rows.append({"block": index, "nll": -sum(logprobs),
                             "avg_nll": -sum(logprobs) / len(logprobs)})
            if phase is not None:
                phase.prepare_next_prefill()
        token_count = len(rows) * args.output_tokens
        total_nll = sum(row["nll"] for row in rows)
        runtime = {"backend": "vllm-v1-teacher-forced-decode",
                   "model": str(args.model), "tokenizer": str(args.tokenizer),
                   "quantization_config": json.loads((args.model / "config.json").read_text()).get("quantization_config", {}),
                   "phase_hetero": args.phase_hetero,
                   "input_tokens": args.input_tokens,
                   "decode_tokens": args.output_tokens,
                   "batch_size": args.batch_size,
                   "max_num_batched_tokens": args.batch_size * required,
                   "gpu_memory_utilization": args.gpu_memory_utilization,
                   "blocks": len(rows), "sparse_bf16_cache_entries": os.environ.get("CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES", "default-512")}
        if phase is not None:
            trace = args.output.with_suffix(".phase_trace.json")
            phase.dump_trace(trace)
            trace_payload = json.loads(trace.read_text())
            event_counts: dict[str, int] = {}
            for event in trace_payload.get("trace", []):
                name = event.get("event")
                if name:
                    event_counts[name] = event_counts.get(name, 0) + 1
            if not event_counts.get("enter_decode") or not event_counts.get("apply_decode"):
                raise RuntimeError(
                    f"{args.label}: phase trace lacks an actual decode transition: {event_counts}"
                )
            runtime["phase_trace"] = str(trace)
            runtime["phase_trace_events"] = event_counts
        result = {"label": args.label, "total_nll": total_nll,
                  "token_count": token_count, "avg_nll": total_nll / token_count,
                  "perplexity": math.exp(total_nll / token_count),
                  "elapsed_seconds": time.perf_counter() - started,
                  "runtime": runtime, "blocks": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("label", "avg_nll", "perplexity", "token_count")}), flush=True)
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
