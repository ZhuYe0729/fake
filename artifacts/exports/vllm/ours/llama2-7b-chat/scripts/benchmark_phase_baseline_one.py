#!/usr/bin/env python3
"""One fresh-process phase-hetero run with the Llama2 baseline workload."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS_ROOT = Path("/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-seq", type=int, default=2048)
    parser.add_argument("--output-seq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, default=VLLM_ROOT)
    parser.add_argument("--cutlass-wrapper-path", type=Path, default=CUTLASS_ROOT)
    args = parser.parse_args()
    configure(args)
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization import phase_hetero_mytest

    max_len = args.input_seq + args.output_seq
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    overrides = {"max_position_embeddings": max_len} if int(getattr(config, "max_position_embeddings", 0) or 0) < max_len else None
    llm = LLM(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1, max_model_len=max_len, max_num_seqs=args.batch, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False, hf_overrides=overrides)
    vocab = int(getattr(llm.get_tokenizer(), "vocab_size", 32000))
    generator = torch.Generator(device="cpu"); generator.manual_seed(args.seed + args.batch * 1000003 + args.input_seq)
    ids = torch.randint(100, max(101, min(vocab, 30000)), (args.batch, args.input_seq), generator=generator, dtype=torch.int64)
    prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in ids]
    sampling = SamplingParams(max_tokens=args.output_seq, min_tokens=args.output_seq, temperature=0.0, ignore_eos=True, detokenize=False)
    phase_hetero_mytest.enable_phase_hetero()
    torch.cuda.synchronize(); started = time.perf_counter(); outputs = llm.generate(prompts, sampling, use_tqdm=False); torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    generated = sum(len(item.outputs[0].token_ids) for item in outputs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"checkpoint": str(args.checkpoint), "batch": args.batch, "input_seq": args.input_seq, "output_seq": args.output_seq, "elapsed_ms": elapsed_ms, "generated_tokens": generated, "timing_scope": "generate_only_after_loaded_llm", "baseline_alignment": "TokensPrompt,max_model_len,max_num_seqs,prefix_cache_disabled,fixed_greedy_output"}, indent=2) + "\n")


def configure(args: argparse.Namespace) -> None:
    sys.path[:0] = [str(args.vllm_root / "vllm"), str(args.vllm_root), str(args.cutlass_wrapper_path)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)


if __name__ == "__main__":
    main()
