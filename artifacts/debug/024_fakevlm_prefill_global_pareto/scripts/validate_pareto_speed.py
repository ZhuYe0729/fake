#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from common_fakevlm_pareto import (
    DEBUG_ROOT,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_JSON,
    append_csv,
    local_cuda_index,
    read_csv,
    read_json,
    write_json,
)
from fakevlm_policy_runtime import apply_policy_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selected FakeVLM Pareto policies with real prefill speed.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--points", default="validation", help="'validation' or comma-separated point indices")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from eval_fakevlm_uniform_accuracy import FakeVLMDataset as AccuracyDataset
    from run_fakevlm_prefill_speed import FakeVLMDataset as SpeedDataset
    from run_fakevlm_prefill_speed import benchmark_prefill, first_batch, load_fakevlm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{local_cuda_index(args.gpu)}")
    torch.cuda.set_device(device)
    policies = select_policies(args)
    out_path = args.output_root / "validation" / "pareto_speed_validation.csv"
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    done = existing_keys(out_path)

    dataset = SpeedDataset(model_path=args.model_path, test_json_file=args.test_json_file, image_root=args.image_root, sample_limit=args.sample_limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    calib_dataset = AccuracyDataset(model_path=args.model_path, test_json_file=args.test_json_file, image_root=args.image_root, sample_limit=max(args.calib_samples, 1))
    calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    batch = first_batch(loader, device=device)

    rows = []
    for item in policies:
        key = f"batch_{args.batch_size}_point_{item['point_index']:03d}"
        if key in done:
            print(f"[skip] existing speed row {key}")
            continue
        policy = read_json(Path(item["policy_json"]))
        model = load_fakevlm(args.model_path, device)
        report = apply_policy_runtime(model, policy, calib_loader=calib_loader, device=device, calib_samples=args.calib_samples)
        result = benchmark_prefill(model, batch, warmup=args.warmup, iters=args.iters)
        row = {
            **item,
            "key": key,
            "actual_batch_size": int(batch["input_ids"].shape[0]),
            "input_tokens": int(batch["input_ids"].shape[1]),
            "e2e_prefill_mean_ms": f"{result.mean_ms:.6f}",
            "e2e_prefill_p50_ms": f"{result.p50_ms:.6f}",
            "e2e_prefill_p90_ms": f"{result.p90_ms:.6f}",
            "e2e_prefill_min_ms": f"{result.min_ms:.6f}",
            "e2e_prefill_max_ms": f"{result.max_ms:.6f}",
            "samples_per_sec": f"{args.batch_size * 1000.0 / result.mean_ms:.6f}",
            "replaced_linear_count": report.replaced_linear_count,
            "skipped_linear_count": report.skipped_linear_count,
            "backend_counts": report.backend_counts,
            "runtime_skipped": report.skipped[:20],
            "warmup": args.warmup,
            "iters": args.iters,
        }
        rows.append(row)
        append_csv(out_path, [row])
        print(f"[done] {key} mean_ms={result.mean_ms:.3f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output_root / "validation" / f"pareto_speed_validation_batch_{args.batch_size}_metadata.json", {"rows_written": len(rows)})


def select_policies(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [row for row in read_csv(args.output_root / "validation" / "selected_pareto_points.csv") if int(float(row["batch_size"])) == args.batch_size]
    if args.points == "validation":
        return normalize(selected)
    wanted = {int(item) for item in args.points.split(",") if item.strip()}
    return normalize([row for row in selected if int(float(row["point_index"])) in wanted])


def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["batch_size"] = int(float(row["batch_size"]))
        item["point_index"] = int(float(row["point_index"]))
        out.append(item)
    return out


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["key"] for row in read_csv(path)}


if __name__ == "__main__":
    main()
