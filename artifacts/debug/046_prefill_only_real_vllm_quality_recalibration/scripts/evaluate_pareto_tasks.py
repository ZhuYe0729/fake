#!/usr/bin/env python3
"""Evaluate one solved Pareto policy on real-vLLM prefill-only tasks.

The policy checkpoint is exported under /tmp for the duration of the process and
removed afterwards.  Task result JSON files are persistent and are skipped on a
later invocation, so an interrupted multi-task evaluation is resumable without
retaining a large checkpoint on disk.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import CUTLASS, MODELS, VLLM_ROOT, model_root, normalized_policy, sha256

TASKS = ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu")
METRIC_PREFIXES = ("acc,", "acc_norm,", "word_perplexity,", "byte_perplexity,", "bits_per_byte,")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", type=int)
    parser.add_argument("--policy-json", type=Path,
                        help="Evaluate an explicit policy instead of a solved point.")
    parser.add_argument("--label", help="Persistent result/checkpoint label for --policy-json.")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temporary-root", type=Path, default=Path("/tmp/cospaq_phase_pareto_046_tasks"))
    parser.add_argument("--experiment-root", type=Path,
                        help="Override the default 046 Llama2 experiment root.")
    parser.add_argument("--canonical-sparse-bf16-state", type=Path)
    parser.add_argument("--canonical-sparse-nvfp4-state", type=Path)
    return parser.parse_args()


def metric_values(payload: dict[str, Any], task: str) -> dict[str, Any]:
    selected = payload.get("results", {}).get(task) or payload.get("groups", {}).get(task) or {}
    return {key: value for key, value in selected.items() if key.startswith(METRIC_PREFIXES)}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def uses_sparse_bf16(policy: dict[str, Any]) -> bool:
    methods = [policy["default_prefill_method"], policy["default_decode_method"]]
    methods.extend(method for pair in policy["method_map"].values() for method in pair.values())
    return "sparse_bf16" in methods


def main() -> None:
    args = parse_args()
    requested = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    unknown = set(requested) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    root = args.experiment_root or model_root("llama2")
    if (args.point is None) == (args.policy_json is None):
        raise ValueError("provide exactly one of --point or --policy-json")
    label = args.label or f"point_{args.point:03d}"
    policy_path = args.policy_json or root / "pareto/policies" / f"{label}.json"
    policy = normalized_policy(policy_path)
    out_root = root / "pareto/validation/tasks" / label
    profile = "full" if args.limit is None else f"limit_{args.limit}"
    missing = [task for task in requested if not (out_root / task / profile / "result.json").exists()]
    if not missing:
        print(f"already complete: {label} {profile}")
        return

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ.setdefault("CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR", str(root / "build/sparse_bf16_tasks"))
    sparse = uses_sparse_bf16(policy)
    if sparse:
        os.environ["CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES"] = "4"
    checkpoint = args.temporary_root / label
    exporter = VLLM_ROOT / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
    export = [sys.executable, str(exporter), "--model-path", str(MODELS["llama2"]["path"]),
              "--policy-json", str(policy_path), "--output-dir", str(checkpoint),
              "--cutlass-wrapper-path", str(CUTLASS)]
    canonical_sparse = args.canonical_sparse_bf16_state or args.canonical_sparse_nvfp4_state
    if args.canonical_sparse_bf16_state:
        export.extend(["--canonical-sparse-bf16-state", str(args.canonical_sparse_bf16_state)])
    if args.canonical_sparse_nvfp4_state:
        export.extend(["--canonical-sparse-nvfp4-state", str(args.canonical_sparse_nvfp4_state)])
    methods = [policy["default_prefill_method"], policy["default_decode_method"]]
    methods.extend(method for pair in policy["method_map"].values() for method in pair.values())
    if any(method.startswith("sparse_") for method in methods) and not canonical_sparse:
        export.append("--prune")
    lm = None
    try:
        import subprocess
        if checkpoint.exists():
            # A prior task chain can be interrupted after exporting its model.
            # Reusing it is safe only when its embedded policy is identical.
            if normalized_policy(checkpoint / "phase_hetero_policy.json") != policy:
                raise RuntimeError(f"stale checkpoint policy differs: {checkpoint}")
            print(f"reusing interrupted temporary checkpoint: {checkpoint}", flush=True)
        else:
            subprocess.run(export, check=True)
        if normalized_policy(checkpoint / "phase_hetero_policy.json") != policy:
            raise RuntimeError("exported policy differs from source JSON")
        sys.path[:0] = [str(VLLM_ROOT / "vllm"), str(VLLM_ROOT), str(CUTLASS)]
        from lm_eval import simple_evaluate
        from lm_eval.models.vllm_causallms import VLLM
        from lm_eval.tasks import TaskManager
        from vllm.model_executor.layers.quantization import phase_hetero_mytest

        lm = VLLM(pretrained=str(checkpoint), tokenizer=str(MODELS["llama2"]["path"]), dtype="bfloat16",
                  batch_size=1 if sparse else 4, max_model_len=2048, tensor_parallel_size=1,
                  enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False,
                  gpu_memory_utilization=0.8 if sparse else 0.9, max_num_seqs=1 if sparse else 4,
                  skip_mm_profiling=True)
        phase_hetero_mytest.enable_phase_hetero()
        runtime = {"backend": "lm_eval.VLLM/vllm-runtime", "checkpoint_temporary": True,
                   "policy_json": str(policy_path), "policy_sha256": sha256(policy_path),
                   "phase_hetero_prefill_enabled": True, "max_model_len": 2048,
                   "gpu_memory_utilization": 0.8 if sparse else 0.9,
                   "sparse_bf16_matmul_cache_entries": os.environ.get("CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES", "default-512"),
                   "vllm": package_version("vllm"), "lm_eval": package_version("lm-eval"),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        for task in missing:
            lm.batch_size = 1 if task == "wikitext" or sparse else 4
            result = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, batch_size=lm.batch_size,
                                     limit=args.limit, log_samples=False, random_seed=0,
                                     numpy_random_seed=0, torch_random_seed=0,
                                     fewshot_random_seed=0, task_manager=TaskManager())
            if result is None:
                raise RuntimeError(f"lm-eval returned None for {task}")
            row = {"policy": label, "task": task, "metrics": metric_values(result, task),
                   "limit": args.limit, "num_fewshot": 0, "batch_size": lm.batch_size,
                   "started_at_utc": started_at, "elapsed_seconds": time.perf_counter() - start,
                   "runtime": runtime, "raw_lm_eval": result}
            target = out_root / task / profile / "result.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
            print(json.dumps({"policy": label, "task": task, "metrics": row["metrics"]}), flush=True)
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
        shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
