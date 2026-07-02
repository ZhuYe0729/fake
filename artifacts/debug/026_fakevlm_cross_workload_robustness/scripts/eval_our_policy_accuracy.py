#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = next(
    (parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "fake").is_dir() and (parent / "artifacts").is_dir()),
    SCRIPT_DIR.parents[4],
)
SOURCE_020_ROOT = REPO_ROOT / "artifacts/debug/020_fakevlm_uniform_accuracy"
for path in (REPO_ROOT, SOURCE_020_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from eval_fakevlm_uniform_accuracy import FakeVLMDataset, validate  # noqa: E402
from run_e2e_speed import apply_our_policy, read_json  # noqa: E402
from fake.compression.modules import select_compressible_modules  # noqa: E402


DEFAULT_MODEL_PATH = "/home/agent/wja/data/models/lingcco/fakeVLM"
DEFAULT_TEST_JSON = "/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json"
DEFAULT_IMAGE_ROOT = "/home/agent/wja/data/datasets/lingcco/FakeClue/test/test"
SCENARIOS = ("prefill_only", "normal_01", "normal_02")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FakeVLM 026 our-linear-hybrid policy accuracy.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.method = "our_linear_hybrid"
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{local_cuda_index(args.gpu)}")
    torch.cuda.set_device(device)

    out_dir = args.output_root / "accuracy" / args.scenario / "our_linear_hybrid"
    accuracy_path = out_dir / "accuracy.json"
    if accuracy_path.exists() and not args.overwrite:
        print(f"[skip] existing accuracy={accuracy_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_path = args.output_root / "policies" / args.scenario / "our_linear_hybrid" / "policy.json"
    policy = read_json(policy_path)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval().to(device)
    model.requires_grad_(False)

    selected = select_compressible_modules(model, "fakevlm")
    selected_linears = [info for info in selected if info.kind == "linear"]

    dataset = FakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=args.sample_limit,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    calib_loader = None
    if policy_needs_calibration(policy):
        calib_dataset = FakeVLMDataset(
            model_path=args.model_path,
            test_json_file=args.test_json_file,
            image_root=args.image_root,
            sample_limit=max(args.calib_samples, 1),
        )
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=args.calib_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    runtime_args = argparse.Namespace(
        method="our_linear_hybrid",
        calib_samples=args.calib_samples,
        calib_batch_size=args.calib_batch_size,
    )
    report = apply_our_policy(runtime_args, model, policy, selected_linears, device, calib_loader)

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.vision_feature_select_strategy = None
    result = validate(args, model, processor, dataloader, device)
    accuracy = result["accuracy"]
    write_json(out_dir / "predictions.json", result["predictions"])
    write_json(accuracy_path, accuracy)
    write_json(out_dir / "runtime_report.json", report)
    write_summary_csv(args.output_root / "accuracy" / "our_linear_hybrid_accuracy.csv", args, accuracy, report, policy_path)
    print(f"[done] scenario={args.scenario} accuracy={accuracy['global_stats']['global_accuracy']:.6f}")


def policy_needs_calibration(policy: dict[str, Any]) -> bool:
    for row in policy.get("modules", []):
        if "sparse_" in str(row.get("selected_prefill_backend")) or "sparse_" in str(row.get("selected_decode_backend")):
            return True
    return False


def write_summary_csv(path: Path, args: argparse.Namespace, accuracy: dict[str, Any], report: dict[str, Any], policy_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario": args.scenario,
        "method": "our_linear_hybrid",
        "global_accuracy": f"{accuracy['global_stats']['global_accuracy']:.6f}",
        "total_right": accuracy["global_stats"]["total_right"],
        "total_wrong": accuracy["global_stats"]["total_wrong"],
        "sample_limit": args.sample_limit or "",
        "batch_size": args.batch_size,
        "calib_samples": args.calib_samples if report.get("backend_counts") else "",
        "replaced_linear_count": report.get("replaced_linear_count", ""),
        "skipped_linear_count": report.get("skipped_linear_count", ""),
        "backend_counts": json.dumps(report.get("backend_counts", {}), sort_keys=True),
        "policy_json": str(policy_path),
    }
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_cuda_index(requested_gpu: int) -> int:
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA is required")
    if requested_gpu < count:
        return requested_gpu
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return 0
    raise RuntimeError(f"requested gpu {requested_gpu}, but torch sees {count} CUDA devices")


if __name__ == "__main__":
    main()
