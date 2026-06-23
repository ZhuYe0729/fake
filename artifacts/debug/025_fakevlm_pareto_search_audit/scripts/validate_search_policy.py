#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from common_search_audit import (
    DEBUG_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_JSON,
    f,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from eval_fakevlm_uniform_accuracy import validate
from fakevlm_policy_runtime import apply_policy_runtime
from run_fakevlm_prefill_speed import benchmark_prefill, first_batch, load_fakevlm


class IndexedFakeVLMDataset(Dataset):
    def __init__(self, *, model_path: str, test_json_file: str, image_root: str, indices: list[int], with_labels: bool) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        self.indices = indices
        self.with_labels = with_labels
        with open(test_json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.vision_feature_select_strategy = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[self.indices[idx]]
        image_path = self.image_root / item["image"]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=item["conversations"][0]["value"],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )
        squeezed = {key: value.squeeze(0) for key, value in inputs.items()}
        if not self.with_labels:
            return squeezed
        return {
            "inputs": squeezed,
            "label": int(item["label"]),
            "image_path": str(image_path),
            "cate": "deepfake",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one FakeVLM search policy with real speed and 20% accuracy.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--key", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--accuracy-batch-size", type=int, default=8)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-subset-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    result_path = args.output_root / "validation" / "policies" / f"{args.key}.json"
    if result_path.exists() and not args.overwrite:
        print(f"[skip] existing key={args.key}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    policy_row = load_policy_row(args.output_root, args.key)
    policy = read_json(Path(policy_row["policy_json"]))
    subset_indices = [int(float(row["source_index"])) for row in read_csv(args.output_root / "subset" / "subset_indices.csv")]
    if args.max_subset_samples is not None:
        subset_indices = subset_indices[: args.max_subset_samples]
    if not subset_indices:
        raise RuntimeError("subset manifest is empty")

    speed_dataset = IndexedFakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        indices=subset_indices[: max(args.batch_size, 1)],
        with_labels=False,
    )
    speed_loader = DataLoader(speed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    quality_dataset = IndexedFakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        indices=subset_indices,
        with_labels=True,
    )
    quality_loader = DataLoader(quality_dataset, batch_size=args.accuracy_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    calib_dataset = IndexedFakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        indices=subset_indices[: max(args.calib_samples, 1)],
        with_labels=True,
    )
    calib_loader = DataLoader(calib_dataset, batch_size=args.calib_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = load_fakevlm(args.model_path, device)
    report = apply_policy_runtime(model, policy, calib_loader=calib_loader, device=device, calib_samples=args.calib_samples)
    speed_batch = first_batch(speed_loader, device=device)
    speed = benchmark_prefill(model, speed_batch, warmup=args.warmup, iters=args.iters)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.vision_feature_select_strategy = None
    quality = validate(arg_view(args), model, processor, quality_loader, device)
    acc = quality["accuracy"]["global_stats"]["global_accuracy"]
    row = {
        **policy_row,
        "actual_batch_size": int(speed_batch["input_ids"].shape[0]),
        "input_tokens": int(speed_batch["input_ids"].shape[1]),
        "e2e_prefill_mean_ms": f"{speed.mean_ms:.6f}",
        "e2e_prefill_p50_ms": f"{speed.p50_ms:.6f}",
        "e2e_prefill_p90_ms": f"{speed.p90_ms:.6f}",
        "e2e_prefill_min_ms": f"{speed.min_ms:.6f}",
        "e2e_prefill_max_ms": f"{speed.max_ms:.6f}",
        "samples_per_sec": f"{args.batch_size * 1000.0 / speed.mean_ms:.6f}",
        "global_accuracy": f"{acc:.8f}",
        "total_right": quality["accuracy"]["global_stats"]["total_right"],
        "total_wrong": quality["accuracy"]["global_stats"]["total_wrong"],
        "subset_samples": len(subset_indices),
        "accuracy_batch_size": args.accuracy_batch_size,
        "calib_samples": args.calib_samples,
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "backend_counts_runtime": report.backend_counts,
        "runtime_skipped": report.skipped[:20],
        "warmup": args.warmup,
        "iters": args.iters,
    }
    write_json(result_path, {"row": row, "accuracy": quality["accuracy"]})
    write_csv(args.output_root / "validation" / "policies" / f"{args.key}.csv", [row])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[done] key={args.key} mean_ms={speed.mean_ms:.3f} acc={acc:.6f}")


def load_policy_row(output_root: Path, key: str) -> dict[str, Any]:
    rows = read_csv(output_root / "search" / "search_policies.csv")
    for row in rows:
        if row["key"] == key:
            return row
    raise KeyError(f"unknown policy key: {key}")


def arg_view(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(method=args.key, max_new_tokens=args.max_new_tokens)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
