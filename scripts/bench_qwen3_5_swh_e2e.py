#!/usr/bin/env python3
"""Real compressed E2E inference: Qwen3.5-9B — all 5 single-kernel methods vs hybrid.

Each method converts the dense model in-memory (no checkpoint save/load),
then runs prefill + decode with KV cache.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="9B")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--input-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--policy-json", default=None, help="Offline policy JSON for predictor_hybrid.")
    p.add_argument("--methods", nargs="+", default=None, help="Methods to benchmark; defaults to all methods.")
    p.add_argument("--output-csv", default=None, help="Optional CSV path for benchmark rows.")
    p.add_argument("--scenario-name", default="custom", help="Scenario label written to --output-csv.")
    return p.parse_args()


def load_dense(variant: str, dtype: torch.dtype):
    from fake.models.qwen3_5 import load_qwen3_5_dense, qwen3_5_model_path
    print(f"  Loading Qwen3.5-{variant} dense model...")
    model_path = str(qwen3_5_model_path(variant))
    # 27B (~54GB bf16) needs multi-GPU; use device_map="auto" to spread across all GPUs
    if variant == "27B":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype,
            local_files_only=True, device_map="auto",
        )
        print(f"  Multi-GPU: {set(p.device for p in model.parameters() if p.device.type == 'cuda')}")
    else:
        model, _config = load_qwen3_5_dense(model_id=model_path, device="cuda", torch_dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def convert_inplace(model: nn.Module, method: str, dtype: torch.dtype, policy_json: str | None = None):
    """Apply kernel replacement to model in-memory."""
    from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config, replace_linear_with_cutlass_nvfp4
    from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config, replace_linear_with_cutlass_sparse_bf16
    from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config, replace_linear_with_cutlass_sparse_nvfp4
    from fake.kernels.marlin_nvfp4 import MarlinNVFP4Config, prepare_marlin_nvfp4_packed_model
    from fake.models.qwen3_5_kernels import replace_linear_with_qwen_predictor_hybrid, replace_linear_with_qwen_swh

    if method == "dense":
        return {"replaced_linear_count": "N/A", "skipped_linear_count": 0}
    if method == "dense_nvfp4":
        report = replace_linear_with_cutlass_nvfp4(model, "qwen3_5", CutlassNVFP4Config())
    elif method == "sparse_bf16":
        report = replace_linear_with_cutlass_sparse_bf16(model, "qwen3_5", CutlassSparseBF16Config(prune=True))
    elif method == "sparse_nvfp4":
        report = replace_linear_with_cutlass_sparse_nvfp4(model, "qwen3_5", CutlassSparseNVFP4Config(prune=True))
    elif method == "marlin_nvfp4":
        _metadata, report = prepare_marlin_nvfp4_packed_model(
            model, "qwen3_5", MarlinNVFP4Config(activation_dtype=dtype),
        )
    elif method == "shape_workload_hybrid":
        report = replace_linear_with_qwen_swh(model, activation_dtype=dtype)
    elif method == "predictor_hybrid":
        if policy_json is None:
            raise ValueError("--policy-json is required for predictor_hybrid")
        report = replace_linear_with_qwen_predictor_hybrid(
            model,
            policy_path=policy_json,
            activation_dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Show skipped modules (marlin can't do N=32, sparse_bf16 can't do certain shapes, etc.)
    skipped = getattr(report, "skipped_linear_count", 0)
    replaced = getattr(report, "replaced_linear_count", 0)
    if skipped:
        skipped_list = getattr(report, "skipped", [])
        for s in skipped_list[:5]:
            print(f"    skipped: {s['name']} — {s.get('reason', '?')}")
    print(f"    replaced={replaced}, skipped={skipped}")

    return report


@torch.inference_mode()
def benchmark(model: nn.Module, batch_size: int, input_tokens: int, output_tokens: int,
              warmup_iters: int) -> dict:
    """Run prefill + decode and return latencies."""
    # Warmup
    for _ in range(warmup_iters):
        ids = torch.randint(0, 1000, (batch_size, 32), device="cuda")
        _ = model(ids)
    torch.cuda.synchronize()

    # Prefill
    prefill_ids = torch.randint(0, 1000, (batch_size, input_tokens), device="cuda")
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    prefill_out = model(prefill_ids, use_cache=True)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = start.elapsed_time(end)

    past_key_values = prefill_out.past_key_values
    next_token = torch.randint(0, 1000, (batch_size, 1), device="cuda")

    decode_times = []
    for _ in range(output_tokens):
        start.record()
        out = model(next_token, past_key_values=past_key_values, use_cache=True)
        end.record()
        torch.cuda.synchronize()
        decode_times.append(start.elapsed_time(end))
        past_key_values = out.past_key_values
        next_token = torch.randint(0, 1000, (batch_size, 1), device="cuda")

    avg_decode = sum(decode_times) / len(decode_times)
    return {
        "prefill_ms": prefill_ms,
        "decode_avg_ms": avg_decode,
        "decode_first_ms": decode_times[0],
        "decode_steady_ms": sum(decode_times[2:]) / max(len(decode_times[2:]), 1),
    }


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    default_methods = [
        "dense",
        "dense_nvfp4",
        "sparse_bf16",
        "sparse_nvfp4",
        "marlin_nvfp4",
        "shape_workload_hybrid",
    ]
    if args.policy_json is not None:
        default_methods.append("predictor_hybrid")
    methods = args.methods or default_methods
    valid_methods = set(default_methods) | {"predictor_hybrid"}
    unknown_methods = sorted(set(methods) - valid_methods)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    if "predictor_hybrid" in methods and args.policy_json is None:
        raise ValueError("--policy-json is required for predictor_hybrid")

    print("=" * 75)
    print("Qwen3.5 Shape-Workload Hybrid — Real E2E Inference")
    print(f"  Variant: {args.variant}, Batch: {args.batch_size}, "
          f"In: {args.input_tokens}, Out: {args.output_tokens}")
    print(f"  M_prefill ≈ {args.batch_size * args.input_tokens}, M_decode = {args.batch_size}")
    print("=" * 75)

    results = {}
    failed_methods = []
    for method in methods:
        print(f"\n{'─'*60}")
        print(f"  Method: {method}")
        print(f"{'─'*60}")

        try:
            model = load_dense(args.variant, dtype)
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            failed_methods.append((method, f"load: {e}"))
            continue

        try:
            report = convert_inplace(model, method, dtype, policy_json=args.policy_json)
        except Exception as e:
            print(f"  CONVERT FAILED: {e}")
            failed_methods.append((method, f"convert: {e}"))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        try:
            result = benchmark(model, args.batch_size, args.input_tokens,
                              args.output_tokens, args.warmup_iters)
        except Exception as e:
            print(f"  BENCHMARK FAILED: {e}")
            failed_methods.append((method, f"benchmark: {e}"))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        result["report"] = report
        results[method] = result

        print(f"  Prefill:    {result['prefill_ms']:.2f}ms")
        print(f"  Decode avg: {result['decode_avg_ms']:.2f}ms/step "
              f"(first={result['decode_first_ms']:.2f}, steady={result['decode_steady_ms']:.2f})")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Comparison ──
    if failed_methods:
        print("\n  ⚠ Failed methods:")
        for name, reason in failed_methods:
            print(f"    {name}: {reason}")

    if "dense" not in results:
        print("dense baseline was not run; skipping speedup comparison")
        write_results_csv(args, results)
        return

    dense = results["dense"]
    dense_e2e = dense["prefill_ms"] + args.output_tokens * dense["decode_avg_ms"]
    hybrid = results.get("shape_workload_hybrid")
    hybrid_e2e = None
    if hybrid is not None:
        hybrid_e2e = hybrid["prefill_ms"] + args.output_tokens * hybrid["decode_avg_ms"]

    print("\n" + "=" * 75)
    print("E2E COMPARISON  (baseline: dense_bf16 = 1.00x, higher = faster)")
    print("=" * 75)
    print(f"{'Method':<24s} {'Prefill':>9s}  {'Dec×n':>9s}  {'E2E':>9s}  {'Speedup':>8s}")
    print("-" * 65)

    best_single_sp = 0.0
    best_single_name = ""

    for method in methods:
        r = results[method]
        e2e = r["prefill_ms"] + args.output_tokens * r["decode_avg_ms"]
        sp = dense_e2e / e2e
        marker = "  ← fastest" if method == "shape_workload_hybrid" and hybrid is not None else ""
        print(f"{method:<24s} {r['prefill_ms']:9.2f}  {r['decode_avg_ms']*args.output_tokens:9.2f}  "
              f"{e2e:9.2f}  {sp:7.4f}x{marker}")
        if method != "shape_workload_hybrid" and sp > best_single_sp:
            best_single_sp = sp
            best_single_name = method

    print(f"\n  dense_bf16  = 1.00x (baseline)")
    if best_single_name:
        print(f"  best single = {best_single_sp:.4f}x ({best_single_name})")
    if hybrid_e2e is not None:
        hybrid_sp = dense_e2e / hybrid_e2e
        print(f"  hybrid      = {hybrid_sp:.4f}x  ← fastest")
        if best_single_sp:
            print(f"  hybrid / {best_single_name} = {hybrid_sp / best_single_sp:.4f}x")

    write_results_csv(args, results)


def write_results_csv(args, results: dict[str, dict]) -> None:
    if args.output_csv is None:
        return
    if not results:
        print(f"No successful results; CSV not written: {args.output_csv}")
        return
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, result in results.items():
        report = result.get("report")
        decode_x_n = result["decode_avg_ms"] * args.output_tokens
        rows.append(
            {
                "model": f"Qwen3.5-{args.variant}",
                "scenario": args.scenario_name,
                "batch_size": args.batch_size,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "method": method,
                "prefill_ms": f"{result['prefill_ms']:.4f}",
                "decode_avg_ms": f"{result['decode_avg_ms']:.4f}",
                "decode_x_n_ms": f"{decode_x_n:.4f}",
                "e2e_ms": f"{result['prefill_ms'] + decode_x_n:.4f}",
                "replaced_linear_count": getattr(report, "replaced_linear_count", ""),
                "skipped_linear_count": getattr(report, "skipped_linear_count", ""),
                "backend_counts": getattr(report, "backend_counts", ""),
                "policy_json": args.policy_json or "",
            }
        )
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote CSV: {output}")


if __name__ == "__main__":
    main()
