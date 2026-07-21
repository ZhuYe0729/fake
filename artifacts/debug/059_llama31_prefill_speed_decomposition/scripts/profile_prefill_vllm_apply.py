#!/usr/bin/env python3
"""Measure real phase-vLLM prefill Linear apply time over two warmed passes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS_ROOT = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-seq", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--skip-explicit-prefill-prepare", action="store_true",
                        help="Reproduce the historical benchmark's phase setup.")
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

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.batch * 1_000_003 + args.input_seq)
    tokens = torch.randint(100, 30_000, (args.batch, args.input_seq),
                           generator=generator, dtype=torch.int64)
    prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in tokens]
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    total_seq = args.input_seq + 1
    overrides = ({"max_position_embeddings": total_seq}
                 if int(getattr(config, "max_position_embeddings", 0) or 0) < total_seq
                 else None)
    llm = LLM(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=total_seq, max_num_seqs=args.batch,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_num_batched_tokens=args.batch * args.input_seq, hf_overrides=overrides)
    sampling = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0,
                              ignore_eos=True, detokenize=False)
    original = PhaseHeteroMyTestLinearMethod.apply
    active_pass: int | None = None
    records: list[tuple[int, str, str, str, int, object, object]] = []

    def wrapped(method, layer, x, bias=None):
        if active_pass is None:
            return original(method, layer, x, bias)
        phase = runtime.current_phase()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = original(method, layer, x, bias)
        end.record()
        selected = layer.ph_prefill_method if phase == "prefill" else layer.ph_decode_method
        records.append((active_pass, phase, selected, layer.prefix,
                        int(x.numel() // layer.input_size_per_partition), start, end))
        return output

    PhaseHeteroMyTestLinearMethod.apply = wrapped
    try:
        runtime.enable_phase_hetero()
        llm.generate(prompts, sampling, use_tqdm=False)  # warm loading and plans
        passes = []
        for pass_index in range(2):
            if not args.skip_explicit_prefill_prepare:
                runtime.prepare_next_prefill()
                runtime.wait_for_prefill_ready()
            active_pass = pass_index
            torch.cuda.synchronize()
            started = time.perf_counter()
            llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            passes.append({"pass": pass_index, "e2e_wall_ms": (time.perf_counter() - started) * 1000.0})
            active_pass = None

        grouped: dict[str, float] = defaultdict(float)
        per_layer: dict[str, float] = defaultdict(float)
        for pass_index, phase, method, prefix, tokens, start, end in records:
            end.synchronize()
            elapsed = start.elapsed_time(end)
            grouped[f"pass{pass_index}/{phase}/{method}"] += elapsed
            per_layer[f"pass{pass_index}/{phase}/{prefix}/{method}/tokens={tokens}"] += elapsed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "batch": args.batch,
            "input_seq": args.input_seq,
            "explicit_prefill_prepare": not args.skip_explicit_prefill_prepare,
            "passes": passes,
            "grouped_apply_cuda_ms": grouped,
            "per_layer_apply_cuda_ms": per_layer,
            "record_count": len(records),
        }, indent=2) + "\n")
    finally:
        PhaseHeteroMyTestLinearMethod.apply = original
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
