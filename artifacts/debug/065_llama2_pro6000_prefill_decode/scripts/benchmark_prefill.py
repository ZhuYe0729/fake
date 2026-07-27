#!/usr/bin/env python3
"""One fresh-process phase-hetero benchmark with in-engine warmup/repeats."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


from common import CUTLASS as CUTLASS_ROOT
from common import PROTOCOL, VLLM_ROOT
from vllm_compat import (assert_chunked_prefill_disabled,
                         force_v1_chunked_prefill_disabled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=PROTOCOL["batch"])
    parser.add_argument("--input-seq", type=int, default=PROTOCOL["input_tokens"])
    parser.add_argument("--output-seq", type=int, default=PROTOCOL["output_tokens"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=PROTOCOL["gpu_memory_utilization"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--vllm-root", type=Path, default=VLLM_ROOT)
    parser.add_argument("--cutlass-wrapper-path", type=Path, default=CUTLASS_ROOT)
    args = parser.parse_args()
    configure(args)
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization import phase_hetero_mytest
    force_v1_chunked_prefill_disabled()

    max_len = args.input_seq + args.output_seq
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    overrides = {"max_position_embeddings": max_len} if int(getattr(config, "max_position_embeddings", 0) or 0) < max_len else None
    llm = LLM(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=max_len, max_num_seqs=args.batch,
              max_num_batched_tokens=args.batch * max_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              hf_overrides=overrides)
    assert_chunked_prefill_disabled(llm)
    vocab = int(getattr(llm.get_tokenizer(), "vocab_size", 32000))
    generator = torch.Generator(device="cpu"); generator.manual_seed(args.seed + args.batch * 1000003 + args.input_seq)
    ids = torch.randint(100, max(101, min(vocab, 30000)), (args.batch, args.input_seq), generator=generator, dtype=torch.int64)
    prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in ids]
    sampling = SamplingParams(max_tokens=args.output_seq, min_tokens=args.output_seq, temperature=0.0, ignore_eos=True, detokenize=False)
    if args.warmup_iters < 0 or args.iters <= 0:
        raise ValueError("warmup-iters must be non-negative and iters must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(0)
    expected = {"apply_prefill": 128, "apply_decode": 128 * (args.output_seq - 1),
                "enter_decode": 1, "prepare_next_prefill": 1}
    total = args.warmup_iters + args.iters
    measured: list[float] = []
    trace_offset = 0
    phase_hetero_mytest.enable_phase_hetero()
    for index in range(total):
        if index:
            # The previous restore is asynchronous. Wait outside the timing
            # interval before submitting the next request to the same engine.
            phase_hetero_mytest.wait_for_prefill_ready()
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        generated = sum(len(item.outputs[0].token_ids) for item in outputs)
        # Restore is intentionally outside the measured generate interval.
        phase_hetero_mytest.prepare_next_prefill()

        trace_dump = args.output_dir / ".cumulative_phase_trace.json"
        phase_hetero_mytest.dump_trace(trace_dump)
        cumulative = json.loads(trace_dump.read_text())
        events = cumulative.get("trace", [])[trace_offset:]
        trace_offset += len(events)
        counts: dict[str, int] = {}
        for event in events:
            name = event.get("event")
            if name:
                counts[name] = counts.get(name, 0) + 1
        if any(counts.get(name, 0) != value for name, value in expected.items()):
            raise RuntimeError(f"phase trace mismatch: expected={expected}, actual={counts}")

        is_warmup = index < args.warmup_iters
        ordinal = index if is_warmup else index - args.warmup_iters
        stem = f"warmup_{ordinal}" if is_warmup else f"measured_{ordinal}"
        if args.warmup_iters == 1 and is_warmup:
            stem = "warmup"
        output_json = args.output_dir / f"{stem}.json"
        trace_path = output_json.with_suffix(".phase_trace.json")
        trace_path.write_text(json.dumps({"phase": cumulative.get("phase"),
                                          "num_layers": cumulative.get("num_layers"),
                                          "num_scheduled": cumulative.get("num_scheduled"),
                                          "trace": events}, indent=2) + "\n")
        output_json.write_text(json.dumps({"checkpoint": str(args.checkpoint), "batch": args.batch,
            "input_seq": args.input_seq, "output_seq": args.output_seq, "elapsed_ms": elapsed_ms,
            "generated_tokens": generated, "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": args.batch * max_len,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_name": properties.name, "cuda_device_uuid": str(getattr(properties, "uuid", "unavailable")),
            "chunked_prefill_enabled": False, "prefix_caching_enabled": False,
            "chunked_prefill_guard": "065_process_local_vllm_0.11_compat", "phase_trace": str(trace_path),
            "phase_trace_events": counts, "timing_scope": "generate_only_after_loaded_llm",
            "single_process_repeats": True, "benchmark_process_id": os.getpid(),
            "iteration": ordinal, "warmup": is_warmup,
            "baseline_alignment": "TokensPrompt,max_model_len,max_num_seqs,prefix_cache_disabled,chunked_prefill_disabled,fixed_greedy_output"}, indent=2) + "\n")
        if not is_warmup:
            measured.append(elapsed_ms)
    trace_dump.unlink(missing_ok=True)
    mean = statistics.mean(measured)
    (args.output_dir.parent / "summary.json").write_text(json.dumps({
        "single_process_repeats": True, "benchmark_process_id": os.getpid(),
        "warmup_iters": args.warmup_iters, "measured_runs": args.iters,
        "measured_elapsed_ms": measured, "median_ms": statistics.median(measured),
        "mean_ms": mean, "stdev_ms": statistics.stdev(measured) if len(measured) > 1 else 0.0,
        "cv": statistics.stdev(measured) / mean if len(measured) > 1 and mean else 0.0,
    }, indent=2) + "\n")


def configure(args: argparse.Namespace) -> None:
    sys.path[:0] = [str(args.vllm_root / "vllm"), str(args.vllm_root), str(args.cutlass_wrapper_path)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["PHASE_HETERO_TRACE"] = "1"
    os.environ["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)


if __name__ == "__main__":
    main()
