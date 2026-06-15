#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from common_dialogsum import (
    DEBUG_ROOT,
    DEFAULT_MODEL_KEY,
    PARETO_ROOT,
    SOURCE_ROOT,
    UNIFORM_METHODS,
    apply_policy_compressed_weights,
    cleanup_cuda,
    cleanup_model,
    convert_policy_to_offline,
    dialogsum_generate_and_score,
    dialogsum_reference_nll,
    dtype_from_arg,
    install_pareto_runtime,
    install_uniform_runtime,
    load_compressed_state_into_model,
    load_dialogsum_split,
    load_eval_model,
    local_cuda_index,
    prepare_tokenizer,
    read_csv,
    read_json,
    replacement_report_dict,
    runtime_report_dict,
    utc_now,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DialogSum ROUGE-L and conditional NLL with real compressed kernels.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--pareto-root", type=Path, default=PARETO_ROOT)
    parser.add_argument("--kind", choices=["pareto", "uniform"], required=True)
    parser.add_argument("--items", required=True, help="Comma-separated point indices for pareto, or methods for uniform.")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nll-batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", default="/home/agent/wja/.cache/huggingface")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-name", default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    dataset = load_dialogsum_split(
        split=args.split,
        limit=args.limit,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    tokenizer = prepare_tokenizer()
    out_csv = args.output_root / "quality" / args.run_name / f"dialogsum_{args.kind}_{sanitize(args.items)}.csv"
    rows = read_existing(out_csv) if args.skip_existing else []
    done = {row["item_id"] for row in rows}
    items = parse_items(args)
    print(f"DialogSum quality kind={args.kind} items={items} samples={len(dataset)} gpu={args.gpu}")
    for item in items:
        item_id = str(item)
        if item_id in done:
            print(f"skipping existing item={item_id}")
            continue
        label, model, runtime_report, compression_meta, policy_meta = build_model_for_item(args, item, device=device, dtype=dtype)
        result_dir = args.output_root / "quality" / args.run_name / args.kind / label
        results_jsonl = result_dir / "results.jsonl"
        print(f"evaluating {args.kind}:{label}")
        rouge = dialogsum_generate_and_score(
            model,
            tokenizer,
            dataset,
            device=device,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            results_jsonl=results_jsonl,
        )
        nll = dialogsum_reference_nll(
            model,
            tokenizer,
            dataset,
            device=device,
            batch_size=args.nll_batch_size,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
        )
        row: dict[str, Any] = {
            "kind": args.kind,
            "item_id": item_id,
            "label": label,
            "num_samples": rouge["num_samples"],
            "nll_samples": nll["num_samples"],
            "conditional_nll": nll["nll"],
            "conditional_ppl": nll["ppl"],
            "nll_tokens": nll["tokens"],
            "rouge1": rouge["rouge"]["rouge1"],
            "rouge2": rouge["rouge"]["rouge2"],
            "rougeL": rouge["rouge"]["rougeL"],
            "results_jsonl": str(results_jsonl),
        }
        if policy_meta:
            row.update(policy_meta)
        rows.append(row)
        write_csv(out_csv, rows)
        write_json(
            result_dir / "summary.json",
            {
                **row,
                "timestamp": utc_now(),
                "dataset": {"name": "knkarthick/dialogsum", "split": args.split, "limit": args.limit},
                "generation": {
                    "prompt_template": "Summarize the following dialogue.\\n\\n{dialogue}\\n\\nSummary:",
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                },
                "compression_metadata": compression_meta,
                "runtime_report": runtime_report_dict(runtime_report),
            },
        )
        cleanup_model(model)
    write_json(
        out_csv.with_suffix(".metadata.json"),
        {
            "timestamp": utc_now(),
            "kind": args.kind,
            "items": items,
            "gpu": args.gpu,
            "dtype": args.dtype,
            "attn": args.attn,
            "dataset": {"name": "knkarthick/dialogsum", "split": args.split, "limit": args.limit, "samples": len(dataset)},
            "source_root": str(args.source_root),
            "pareto_root": str(args.pareto_root),
            "metric_note": "conditional_nll is computed only on reference summary tokens; prompt and padding tokens are ignored.",
        },
    )
    print(f"wrote {len(rows)} rows to {out_csv}")


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return read_csv(path)


def parse_items(args: argparse.Namespace) -> list[int | str]:
    if args.kind == "pareto":
        return [int(item) for item in args.items.split(",") if item.strip()]
    methods = [item.strip() for item in args.items.split(",") if item.strip()]
    unknown = [method for method in methods if method not in UNIFORM_METHODS]
    if unknown:
        raise ValueError(f"unknown uniform methods: {unknown}")
    return methods


def build_model_for_item(args: argparse.Namespace, item: int | str, *, device: str, dtype: torch.dtype):
    model = load_eval_model(dtype=dtype, device=device, attn=args.attn)
    compression_meta: dict[str, Any] | None = None
    policy_meta: dict[str, Any] = {}
    if args.kind == "uniform":
        method = str(item)
        compression_meta = load_compressed_state_into_model(
            model,
            method=method,
            source_root=args.source_root,
            model_key=DEFAULT_MODEL_KEY,
            device=device,
        )
        runtime_report = install_uniform_runtime(model, method=method, model_key=DEFAULT_MODEL_KEY, dtype=dtype)
        return method, model, runtime_report, compression_meta, policy_meta

    point = int(item)
    policy_path = find_policy_json(args.pareto_root, point)
    policy = read_json(policy_path)
    compression_meta = apply_policy_compressed_weights(model, policy, source_root=args.source_root)
    converted_path = args.output_root / "quality" / args.run_name / "converted_policies" / f"point_{point:03d}_offline_hybrid_policy.json"
    convert_policy_to_offline(policy_path, converted_path)
    runtime_report = install_pareto_runtime(model, converted_policy_path=converted_path, dtype=dtype)
    policy_meta = {
        "point_index": point,
        "policy_json": str(policy_path),
        "converted_policy_json": str(converted_path),
        "quality_cost": policy["summary"]["quality_cost"],
        "predicted_latency_ms": policy["summary"]["latency_ms"],
        "replaced_weight_modules": compression_meta["replaced_weight_modules"],
    }
    return f"point_{point:03d}", model, runtime_report, compression_meta, policy_meta


def find_policy_json(root: Path, point_index: int) -> Path:
    matches = sorted((root / "pareto" / "policies").glob(f"point_{point_index:03d}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no policy json for point {point_index} under {root}")
    return matches[0]


def sanitize(value: str) -> str:
    return value.replace(",", "_").replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        cleanup_cuda()
