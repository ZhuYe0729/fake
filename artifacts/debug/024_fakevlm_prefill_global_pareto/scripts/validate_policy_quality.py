#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
    write_csv,
    write_json,
)
from fakevlm_policy_runtime import apply_policy_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FakeVLM policy quality on FakeClue.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--policies", choices=["stratified", "validation"], default="stratified")
    parser.add_argument("--points", default="all", help="For validation mode: all or comma-separated batch:point pairs, e.g. 16:0,16:5")
    parser.add_argument("--policy-indices", default=None, help="For stratified mode: comma-separated policy indices to run.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-policies", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoProcessor
    from eval_fakevlm_uniform_accuracy import FakeVLMDataset, validate
    from run_fakevlm_prefill_speed import load_fakevlm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{local_cuda_index(args.gpu)}")
    torch.cuda.set_device(device)
    policies = select_policies(args)
    if args.max_policies is not None:
        policies = policies[: args.max_policies]
    if not policies:
        raise RuntimeError("no policies selected")

    dataset = FakeVLMDataset(model_path=args.model_path, test_json_file=args.test_json_file, image_root=args.image_root, sample_limit=args.sample_limit)
    calib_dataset = FakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=max(args.calib_samples, 1),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    calib_loader = DataLoader(calib_dataset, batch_size=args.calib_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.vision_feature_select_strategy = None

    out_path = args.output_root / "quality" / f"{args.policies}_quality.csv"
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    done = existing_keys(out_path)
    rows = []
    for item in policies:
        key = item["key"]
        if key in done:
            print(f"[skip] existing quality row {key}")
            continue
        policy = read_json(Path(item["policy_json"]))
        model = load_fakevlm(args.model_path, device)
        report = apply_policy_runtime(model, policy, calib_loader=calib_loader, device=device, calib_samples=args.calib_samples)
        result = validate(arg_view(args), model, processor, dataloader, device)
        acc = result["accuracy"]["global_stats"]["global_accuracy"]
        row = {
            **item,
            "global_accuracy": f"{acc:.8f}",
            "total_right": result["accuracy"]["global_stats"]["total_right"],
            "total_wrong": result["accuracy"]["global_stats"]["total_wrong"],
            "replaced_linear_count": report.replaced_linear_count,
            "skipped_linear_count": report.skipped_linear_count,
            "backend_counts": report.backend_counts,
            "runtime_skipped": report.skipped[:20],
            "sample_limit": args.sample_limit if args.sample_limit is not None else "",
            "batch_size": args.batch_size,
            "calib_samples": args.calib_samples,
        }
        rows.append(row)
        append_csv(out_path, [row])
        point_dir = args.output_root / "quality" / args.policies
        write_json(point_dir / f"{key}.json", {"row": row, "accuracy": result["accuracy"]})
        print(f"[done] {key} accuracy={acc:.6f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if rows:
        write_json(args.output_root / "quality" / f"{args.policies}_quality_metadata.json", {"rows_written": len(rows), "policies": len(policies)})
    print(f"validated {len(rows)} policies")


def select_policies(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.policies == "stratified":
        rows = read_csv(args.output_root / "stratified" / "quality_policies.csv")
        wanted = None
        if args.policy_indices:
            wanted = {int(item) for item in args.policy_indices.split(",") if item.strip()}
            rows = [row for row in rows if int(float(row["policy_index"])) in wanted]
        return [
            {
                "key": f"policy_{int(float(row['policy_index'])):03d}",
                "policy_index": int(float(row["policy_index"])),
                "batch_size_for_policy": "",
                "point_index": "",
                "policy_json": row["policy_json"],
                "label": row.get("label", ""),
            }
            for row in rows
        ]
    selected = read_csv(args.output_root / "validation" / "selected_pareto_points.csv")
    wanted = None
    if args.points != "all":
        wanted = {tuple(int(part) for part in spec.split(":", 1)) for spec in args.points.split(",") if spec.strip()}
    out = []
    for row in selected:
        batch = int(float(row["batch_size"]))
        point = int(float(row["point_index"]))
        if wanted is not None and (batch, point) not in wanted:
            continue
        out.append(
            {
                "key": f"batch_{batch}_point_{point:03d}",
                "policy_index": "",
                "batch_size_for_policy": batch,
                "point_index": point,
                "policy_json": row["policy_json"],
                "label": row.get("selection_reason", ""),
            }
        )
    return out


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["key"] for row in read_csv(path)}


def arg_view(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(method="policy", max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
