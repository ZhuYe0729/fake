#!/usr/bin/env python3
"""Run all Llama2 prefill-only likelihood tasks through real vLLM runtime."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality"
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
TASKS = ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu")
METRIC_PREFIXES = ("acc,", "acc_norm,", "word_perplexity,", "byte_perplexity,", "bits_per_byte,")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEBUG / "manifest/policies.json")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-root", type=Path, default=DEBUG / "results")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args()


def setup_runtime() -> None:
    sys.path[:0] = [str(VLLM_ROOT / "vllm"), str(VLLM_ROOT), str(CUTLASS)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ.setdefault("CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR", str(DEBUG / "build/sparse_bf16_runtime_cache_cap"))


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def metrics(payload: dict[str, Any], task: str) -> dict[str, Any]:
    selected = payload.get("results", {}).get(task) or payload.get("groups", {}).get(task) or {}
    return {key: value for key, value in selected.items() if key.startswith(METRIC_PREFIXES)}


def audit_probe(lm: Any) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    token_ids = lm.tok_encode("The capital of France is Paris.")
    output = lm.model.generate(
        [TokensPrompt(prompt_token_ids=token_ids)],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False),
        use_tqdm=False,
    )[0]
    values = []
    for token, info in zip(output.prompt_token_ids or [], output.prompt_logprobs or [], strict=True):
        if info is not None and token in info:
            values.append({"token_id": token, "logprob": float(getattr(info[token], "logprob", info[token]))})
    return {"prompt_token_ids": token_ids, "returned_token_logprobs": values}


def uses_sparse_bf16(entry: dict[str, Any]) -> bool:
    """Identify sparse-BF16 in uniform or phase-heterogeneous policies."""
    if entry["label"] == "sparse_bf16":
        return True
    policy_path = entry.get("policy_json")
    if not policy_path:
        return False
    policy = json.loads((ROOT / policy_path).read_text())
    methods = [policy["default_prefill_method"], policy["default_decode_method"]]
    methods.extend(method for pair in policy["method_map"].values()
                   for method in pair.values())
    return "sparse_bf16" in methods


def main() -> None:
    args = parse_args()
    setup_runtime()
    manifest = json.loads(args.manifest.read_text())
    entry = next((item for item in manifest["policies"] if item["label"] == args.policy), None)
    if entry is None:
        raise KeyError(args.policy)
    checkpoint = ROOT / entry["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    sparse_workspace_limited = uses_sparse_bf16(entry)
    safe_mode = os.environ.get("COSPAQ_TASK_SAFE_MODE") == "1"
    constrained = sparse_workspace_limited or safe_mode
    if constrained:
        # lm-eval's prompt-logprob path briefly materializes a full-vocabulary
        # FP32 tensor. Keep sparse workspaces below the remaining headroom on
        # 32-GB GPUs; the policy and kernel implementation are unchanged.
        os.environ["CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES"] = "4"
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager
    from vllm.model_executor.layers.quantization import phase_hetero_mytest

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    lm = None
    try:
        lm = VLLM(
            pretrained=str(checkpoint), tokenizer=manifest["model_path"], dtype="bfloat16",
            batch_size=1 if constrained else 4, max_model_len=2048, tensor_parallel_size=1,
            enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False,
            # Reserve prompt-logprob headroom for policies that retain sparse
            # kernel workspaces. The evaluation does not need a large KV cache.
            gpu_memory_utilization=0.75 if safe_mode else (0.8 if sparse_workspace_limited else 0.9),
            max_num_seqs=1 if constrained else 4, skip_mm_profiling=True,
        )
        if entry["kind"] == "ours":
            phase_hetero_mytest.enable_phase_hetero()
        policy_root = args.output_root / entry["label"]
        policy_root.mkdir(parents=True, exist_ok=True)
        runtime = {
            "backend": "lm_eval.VLLM/vllm-runtime", "checkpoint": str(checkpoint),
            "kind": entry["kind"], "quantization_config": json.loads((checkpoint / "config.json").read_text()).get("quantization_config", {}),
            "phase_hetero_prefill_enabled": bool(entry["kind"] == "ours"),
            "sparse_bf16_matmul_cache_entries": os.environ.get("CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES", "default-512"),
            "max_model_len": 2048, "gpu_memory_utilization": 0.75 if safe_mode else (0.8 if sparse_workspace_limited else 0.9),
            "safe_mode": safe_mode,
            "vllm_root": str(VLLM_ROOT), "cutlass_wrapper": str(CUTLASS),
            "environment": {"python": platform.python_version(), "lm_eval": version("lm-eval"),
                            "vllm": version("vllm"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        }
        if "policy_sha256" in entry:
            runtime["policy_json"] = entry["policy_json"]
            runtime["policy_sha256"] = entry["policy_sha256"]
        if args.audit:
            runtime["fixed_prompt_logprob_probe"] = audit_probe(lm)
        (policy_root / "runtime.json").write_text(json.dumps(runtime, indent=2, default=str) + "\n")
        for task in TASKS:
            lm.batch_size = 1 if task == "wikitext" or constrained else 4
            result = simple_evaluate(
                model=lm, tasks=[task], num_fewshot=0, batch_size=lm.batch_size,
                limit=args.limit, log_samples=False, random_seed=0, numpy_random_seed=0,
                torch_random_seed=0, fewshot_random_seed=0, task_manager=TaskManager(),
            )
            if result is None:
                raise RuntimeError(f"lm-eval returned None for {task}")
            row = {"policy": entry["label"], "task": task, "metrics": metrics(result, task),
                   "limit": args.limit, "num_fewshot": 0, "batch_size": lm.batch_size,
                   "started_at_utc": started_at, "elapsed_seconds": time.perf_counter() - started,
                   "runtime": runtime, "raw_lm_eval": result}
            target = policy_root / task / ("full" if args.limit is None else f"limit_{args.limit}") / "result.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
            print(json.dumps({"policy": entry["label"], "task": task, "metrics": row["metrics"]}, default=str), flush=True)
    finally:
        if lm is not None:
            try:
                lm.model.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            del lm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
