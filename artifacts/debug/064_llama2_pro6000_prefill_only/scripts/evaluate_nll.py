#!/usr/bin/env python3
"""Score fixed full-prefill token likelihood directly through vLLM."""
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

from common import CUTLASS, VLLM_ROOT, normalized_policy, sha256
from vllm_compat import (assert_chunked_prefill_disabled,
                         force_v1_chunked_prefill_disabled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--phase-hetero", action="store_true")
    parser.add_argument("--blocks", type=int, default=100)
    return parser.parse_args()


def has_sparse_bf16(checkpoint: Path) -> bool:
    policy_path = checkpoint / "phase_hetero_policy.json"
    if policy_path.exists():
        policy = normalized_policy(policy_path)
        methods = [policy["default_prefill_method"], policy["default_decode_method"]]
        methods.extend(value for pair in policy["method_map"].values() for key, value in pair.items() if key.endswith("method"))
        return "sparse_bf16" in methods
    return checkpoint.name == "uniform_sparse_bf16"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.phase_hetero and args.policy_json is None:
        raise ValueError("--phase-hetero requires --policy-json")
    sys.path[:0] = [str(VLLM_ROOT / "vllm"), str(VLLM_ROOT), str(CUTLASS)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    sparse = has_sparse_bf16(args.checkpoint)
    if sparse:
        os.environ["CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES"] = "4"
    if args.phase_hetero:
        os.environ["PHASE_HETERO_TRACE"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    force_v1_chunked_prefill_disabled()
    phase = None
    if args.phase_hetero:
        from vllm.model_executor.layers.quantization import phase_hetero_mytest as phase

    blocks = torch.load(args.samples, map_location="cpu")[: args.blocks]
    if blocks.ndim != 2 or blocks.shape[1] < 2:
        raise ValueError(f"invalid sample tensor: {tuple(blocks.shape)}")
    llm = None
    started = time.perf_counter()
    rows = []
    try:
        # Prompt logprobs transiently retain a vocabulary-sized tensor for a
        # long prompt.  Reserve headroom by limiting KV cache allocation; this
        # changes no model computation or NLL semantics.
        # Recent vLLM validates that the scheduler token budget can admit the
        # configured maximum sequence length.  Keep the original small
        # headroom while making both values consistent.
        max_len = int(blocks.shape[1]) + 8
        llm = LLM(model=str(args.checkpoint), tokenizer=str(args.tokenizer), dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False, skip_mm_profiling=True, max_model_len=max_len, max_num_seqs=1, max_num_batched_tokens=max_len, gpu_memory_utilization=float(os.environ.get("COSPAQ_VLLM_NLL_GPU_MEMORY_UTILIZATION", "0.70")))
        assert_chunked_prefill_disabled(llm)
        if phase is not None:
            phase.enable_phase_hetero()
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        for index, block in enumerate(blocks):
            if phase is not None and index:
                phase.prepare_next_prefill()
                phase.wait_for_prefill_ready()
            ids = block.tolist()
            output = llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)[0]
            returned_ids, returned_logprobs = output.prompt_token_ids or [], output.prompt_logprobs or []
            if len(returned_ids) != len(ids) or len(returned_logprobs) != len(ids):
                raise RuntimeError(f"{args.label}/block {index}: prompt-logprob length mismatch")
            logprobs = []
            for token, values in zip(returned_ids[1:], returned_logprobs[1:], strict=True):
                if values is None or token not in values:
                    raise RuntimeError(f"{args.label}/block {index}: missing target prompt logprob")
                value = values[token]
                logprobs.append(float(getattr(value, "logprob", value)))
            rows.append({"block": index, "token_count": len(logprobs), "nll": -sum(logprobs), "avg_nll": -sum(logprobs) / len(logprobs)})
        token_count = sum(row["token_count"] for row in rows)
        total_nll = sum(row["nll"] for row in rows)
        runtime = {"backend": "vllm-direct-prompt-logprob", "checkpoint": str(args.checkpoint), "checkpoint_config": json.loads((args.checkpoint / "config.json").read_text()).get("quantization_config", {}), "phase_hetero": args.phase_hetero, "sample_sha256": sha256(args.samples), "sparse_bf16_cache_entries": os.environ.get("CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES", "default-512"), "chunked_prefill_enabled": False, "chunked_prefill_guard": "064_process_local_vllm_0.11_compat"}
        if args.policy_json is not None:
            runtime["policy_json"] = str(args.policy_json)
            runtime["policy_sha256"] = sha256(args.policy_json)
        if phase is not None:
            trace_path = args.output.with_suffix(".phase_trace.json")
            phase.dump_trace(trace_path)
            trace = json.loads(trace_path.read_text())
            counts: dict[str, int] = {}
            for event in trace.get("trace", []):
                if event.get("event"):
                    counts[event["event"]] = counts.get(event["event"], 0) + 1
            expected = {
                "apply_prefill": len(blocks) * 128,
                "apply_decode": 0,
                "enter_decode": len(blocks),
                "prepare_next_prefill": max(len(blocks) - 1, 0),
            }
            if any(counts.get(event, 0) != count for event, count in expected.items()):
                raise RuntimeError(
                    f"{args.label}: prefill-only phase trace mismatch: "
                    f"expected={expected}, actual={counts}")
            runtime["phase_trace"] = str(trace_path)
            runtime["phase_trace_events"] = counts
            runtime["prefill_only_phase_audit"] = expected
        result = {"label": args.label, "total_nll": total_nll, "token_count": token_count, "avg_nll": total_nll / token_count, "perplexity": math.exp(total_nll / token_count), "elapsed_seconds": time.perf_counter() - started, "runtime": runtime, "blocks": rows}
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
