#!/usr/bin/env python3
"""Profile one warmed phase-hetero request by Linear method with CUDA events."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS_ROOT = Path("/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--input-seq", type=int, default=2048)
    parser.add_argument("--output-seq", type=int, default=80)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.8)
    args = parser.parse_args()
    sys.path[:0] = [str(VLLM_ROOT / "vllm"), str(VLLM_ROOT), str(CUTLASS_ROOT)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS_ROOT)

    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization import phase_hetero_mytest as runtime
    from vllm.model_executor.layers.quantization.phase_hetero_mytest import PhaseHeteroMyTestLinearMethod

    generator = torch.Generator(device="cpu"); generator.manual_seed(args.batch * 1000003 + args.input_seq)
    tokens = torch.randint(100, 30000, (args.batch, args.input_seq), generator=generator, dtype=torch.int64)
    prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in tokens]
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    total_seq = args.input_seq + args.output_seq
    overrides = {"max_position_embeddings": total_seq} if int(getattr(config, "max_position_embeddings", 0) or 0) < total_seq else None
    llm_kwargs = dict(model=str(args.checkpoint), dtype="bfloat16",
                      tensor_parallel_size=1, max_model_len=total_seq,
                      max_num_seqs=args.batch, gpu_memory_utilization=args.gpu_memory_utilization,
                      enforce_eager=True, enable_prefix_caching=False,
                      enable_chunked_prefill=False,
                      max_num_batched_tokens=args.batch * args.input_seq, hf_overrides=overrides)
    if args.kv_cache_dtype is not None:
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(max_tokens=args.output_seq, min_tokens=args.output_seq, temperature=0.0,
                              ignore_eos=True, detokenize=False)
    original = PhaseHeteroMyTestLinearMethod.apply
    records: list[tuple[str, str, str, int, object, object]] = []
    active = False

    def wrapped(method, layer, x, bias=None):
        if not active:
            return original(method, layer, x, bias)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        phase = runtime.current_phase()
        start.record()
        output = original(method, layer, x, bias)
        end.record()
        records.append((phase,
                        layer.ph_prefill_method if phase == "prefill" else layer.ph_decode_method,
                        layer.prefix,
                        int(x.numel() // layer.input_size_per_partition),
                        start, end))
        return output

    PhaseHeteroMyTestLinearMethod.apply = wrapped
    try:
        runtime.enable_phase_hetero()
        llm.generate(prompts, sampling, use_tqdm=False)  # warm sparse plans
        runtime.prepare_next_prefill(); runtime.wait_for_prefill_ready()
        active = True
        torch.cuda.synchronize(); started = time.perf_counter()
        llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize(); e2e_ms = (time.perf_counter() - started) * 1000.0
        grouped: dict[str, float] = defaultdict(float)
        layers: dict[str, float] = defaultdict(float)
        phase_tokens: dict[str, int] = defaultdict(int)
        for phase, method, prefix, tokens, start, end in records:
            end.synchronize()
            elapsed = start.elapsed_time(end)
            grouped[f"{phase}/{method}"] += elapsed
            layers[f"{phase}/{prefix}/{method}"] += elapsed
            phase_tokens[f"{phase}/tokens={tokens}"] += 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "batch": args.batch,
                                           "input_seq": args.input_seq, "output_seq": args.output_seq,
                                           "gpu_memory_utilization": args.gpu_memory_utilization, "e2e_ms": e2e_ms,
                                           "grouped_cuda_ms": grouped, "layer_cuda_ms": layers,
                                           "phase_token_call_counts": phase_tokens}, indent=2) + "\n")
    finally:
        PhaseHeteroMyTestLinearMethod.apply = original
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
