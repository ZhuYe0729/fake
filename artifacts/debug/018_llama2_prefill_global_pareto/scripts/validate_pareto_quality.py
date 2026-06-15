#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lm_eval
import torch

from common_pareto import DEBUG_ROOT, SOURCE_003_ROOT, f, read_csv, read_json, write_csv, write_json
from real_policy_runtime import apply_real_policy_runtime

QUALITY_SCRIPTS = DEBUG_ROOT.parent / "007_llama2_quality_modeling" / "scripts"
if str(QUALITY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(QUALITY_SCRIPTS))

from common_quality import (  # type: ignore  # noqa: E402
    QualityConfig,
    cleanup_cuda,
    compressible_modules,
    compute_nll,
    dtype_from_arg,
    load_calibration_blocks,
    load_llama_for_quality,
    load_prepared_state,
    local_cuda_index,
    model_spec,
    module_parent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selected Pareto policies with real compressed weights.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_003_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--task", default="arc_challenge")
    parser.add_argument("--arc-limit", type=int, default=128)
    parser.add_argument("--full-arc", action="store_true")
    parser.add_argument("--arc-batch-size", default="8")
    parser.add_argument("--skip-arc", action="store_true")
    parser.add_argument("--points", default="validation", help="'validation' or comma-separated point indices.")
    parser.add_argument("--policy-dir", type=Path, default=None)
    parser.add_argument("--point-output-only", action="store_true")
    parser.add_argument("--quality-subdir", default="quality_points")
    parser.add_argument("--quality-output-name", default="pareto_quality_validation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    config = QualityConfig(
        calib_samples=args.calib_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        source_root=args.source_root,
        output_root=args.output_root,
    )
    blocks, calib_metadata = load_calibration_blocks(config)
    point_indices = select_point_indices(args)
    quality_dir = args.output_root / "validation" / args.quality_subdir
    quality_dir.mkdir(parents=True, exist_ok=True)
    arc_limit = None if args.full_arc else args.arc_limit
    rows = []
    for point_index in point_indices:
        policy_path = find_policy_json(args.output_root, point_index, args.policy_dir)
        print(f"validating point={point_index} policy={policy_path}")
        policy = read_json(policy_path)
        model = load_llama_for_quality(device=device, dtype=dtype)
        report = apply_real_policy_runtime(model, policy, prepared_root=args.source_root / "prepared", activation_dtype=dtype)
        nll = compute_nll(model, blocks, device=device, batch_size=args.batch_size)
        row: dict[str, Any] = {
            "point_index": point_index,
            "policy_json": str(policy_path),
            "replaced_modules": report.replaced_linear_count,
            "skipped_modules": report.skipped_linear_count,
            "backend_counts": dict(report.backend_counts),
            "runtime_skipped": report.skipped,
            "quality_cost": policy["summary"]["quality_cost"],
            "predicted_latency_ms": policy["summary"]["latency_ms"],
            "nll": nll["nll"],
            "ppl": nll["ppl"],
            "tokens": nll["tokens"],
        }
        if not args.skip_arc:
            arc = run_arc_eval(model, task=args.task, dtype=dtype, device=device, batch_size=args.arc_batch_size, limit=arc_limit)
            row.update(
                {
                    "task": args.task,
                    "arc_limit": arc_limit,
                    "arc_acc": arc.get("acc,none"),
                    "arc_acc_norm": arc.get("acc_norm,none"),
                    "arc_sample_len": arc.get("sample_len"),
                }
            )
        rows.append(row)
        write_csv(quality_dir / f"point_{point_index:03d}.csv", [row])
        if not args.point_output_only:
            write_csv(args.output_root / "validation" / args.quality_output_name, rows)
        del model
        cleanup_cuda()
    if not args.point_output_only:
        write_json(
            args.output_root / "validation" / "pareto_quality_validation_metadata.json",
            {
                "source_root": str(args.source_root),
                "gpu": args.gpu,
                "dtype": args.dtype,
                "task": None if args.skip_arc else args.task,
                "arc_limit": None if args.skip_arc else arc_limit,
                "arc_batch_size": None if args.skip_arc else args.arc_batch_size,
                "calibration": calib_metadata,
                "points": point_indices,
            },
        )
    print(f"wrote {len(rows)} validation rows")


def select_point_indices(args: argparse.Namespace) -> list[int]:
    if args.points == "validation":
        selected = read_csv(args.output_root / "validation" / "selected_pareto_points.csv")
        return [int(f(row, "point_index")) for row in selected]
    return [int(item) for item in args.points.split(",") if item.strip()]


def find_policy_json(output_root: Path, point_index: int, policy_dir: Path | None = None) -> Path:
    root = policy_dir if policy_dir is not None else output_root / "pareto" / "policies"
    matches = sorted(root.glob(f"point_{point_index:03d}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no policy json for point {point_index}")
    return matches[0]


def run_arc_eval(model, *, task: str, dtype: torch.dtype, device: str, batch_size: str, limit: int | None) -> dict[str, Any]:
    from lm_eval.models.huggingface import HFLM

    spec = model_spec("llama2-7b")
    lm = HFLM(
        pretrained=model,
        tokenizer=spec["path"],
        backend="causal",
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        trust_remote_code=bool(spec["trust_remote_code"]),
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate returned None")
    return dict(results["results"][task])


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda()
