#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[3]
CUTLASS_WRAPPER_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
for path in (REPO_ROOT, CUTLASS_WRAPPER_ROOT, CUTLASS_WRAPPER_ROOT / "modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from fake.compression.modules import flatten_weight, restore_weight_shape, select_compressible_modules
from fake.compression.pruning import prune_dense_2_4, prune_nvfp4_pair_2_4
from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config, replace_linear_with_cutlass_nvfp4
from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config, replace_linear_with_cutlass_sparse_bf16
from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config, replace_linear_with_cutlass_sparse_nvfp4
from fake.kernels.marlin_nvfp4 import MarlinNVFP4Config, replace_linear_with_marlin_nvfp4
from fake.models.qwen3_5_kernels import QwenHybridDenseNVFP4Linear, _load_wrapper


METHODS = (
    "dense_bf16",
    "sparse_bf16",
    "dense_nvfp4",
    "sparse_nvfp4",
    "marlin_weight_only",
    "dense_nvfp4_prefill_marlin_decode",
)
DEFAULT_MODEL_PATH = "/home/agent/wja/data/models/lingcco/fakeVLM"
DEFAULT_TEST_JSON = "/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json"
DEFAULT_IMAGE_ROOT = "/home/agent/wja/data/datasets/lingcco/FakeClue/test/test"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/debug/020_fakevlm_uniform_accuracy"


class FakeVLMDataset(Dataset):
    def __init__(
        self,
        *,
        model_path: str,
        test_json_file: str,
        image_root: str,
        sample_limit: int | None = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.image_root = Path(image_root)
        with open(test_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data[:sample_limit] if sample_limit is not None else data
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.vision_feature_select_strategy = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
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
        return {
            "inputs": squeezed,
            "label": int(item["label"]),
            "image_path": str(image_path),
            "cate": "deepfake",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FakeVLM uniform runtime compression accuracy.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--decode-m-threshold", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FakeVLM runtime compression evaluation.")

    method_root = args.output_root / "outputs" / args.method
    compression_root = args.output_root / "compression" / args.method
    config_root = args.output_root / "configs"
    method_root.mkdir(parents=True, exist_ok=True)
    compression_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    accuracy_path = method_root / "accuracy.json"
    if accuracy_path.exists() and not args.overwrite:
        print(f"skip existing method={args.method} accuracy={accuracy_path}")
        return

    device = torch.device(args.device)
    write_json(config_root / f"{args.method}_run_config.json", run_config(args))

    print(f"[load] model={args.model_path} method={args.method}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval().to(device)

    dataset = FakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=args.sample_limit,
    )
    calib_dataset = FakeVLMDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=max(args.calib_samples, 1),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.calib_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    selected = select_compressible_modules(model, "fakevlm")
    selected_summary = [
        {
            "name": info.name,
            "kind": info.kind,
            "columns": info.columns,
            "reason": info.reason,
            "out_features": int(getattr(info.module, "out_features", 0)),
        }
        for info in selected
    ]
    write_json(compression_root / "selected_modules.json", selected_summary)
    print(f"[modules] selected={len(selected)}")
    if args.method != "dense_bf16" and not selected:
        raise RuntimeError("No FakeVLM language linear modules were selected for compression.")

    prepare_started = time.time()
    prep = prepare_runtime(model, selected, calib_loader, device, args)
    prep["elapsed_sec"] = round(time.time() - prepare_started, 3)
    write_json(compression_root / "prepare_metadata.json", prep)
    print(f"[prepare] {json.dumps(prep, sort_keys=True)}")
    if args.method != "dense_bf16" and int(prep.get("replaced_linear_count", 0)) <= 0:
        raise RuntimeError(f"Method {args.method} did not replace any modules.")

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.vision_feature_select_strategy = None
    result = validate(args, model, processor, dataloader, device)
    write_json(method_root / "predictions.json", result["predictions"])
    write_json(accuracy_path, result["accuracy"])
    write_summary_csv(method_root / "accuracy.csv", args.method, result["accuracy"], prep)
    print(f"[done] method={args.method} accuracy={result['accuracy']['global_stats']['global_accuracy']:.6f}")


def prepare_runtime(
    model: nn.Module,
    selected: list[Any],
    calib_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.method == "dense_bf16":
        return {
            "method": args.method,
            "replacement_backend": "torch_bf16",
            "replaced_linear_count": 0,
            "skipped_linear_count": 0,
            "calibration_used": False,
            "activation_quant": "none",
        }
    if args.method == "sparse_bf16":
        prune_report = apply_calibrated_pruning(
            model,
            selected,
            calib_loader,
            device,
            args,
            pattern="dense_2_4",
        )
        report = replace_linear_with_cutlass_sparse_bf16(
            model,
            "fakevlm",
            CutlassSparseBF16Config(prune=False),
        )
        return report_metadata(args.method, report, prune_report, "none")
    if args.method == "dense_nvfp4":
        report = replace_linear_with_cutlass_nvfp4(model, "fakevlm", CutlassNVFP4Config())
        return report_metadata(args.method, report, None, "dynamic_tensor_global_scale_online")
    if args.method == "sparse_nvfp4":
        prune_report = apply_calibrated_pruning(
            model,
            selected,
            calib_loader,
            device,
            args,
            pattern="nvfp4_pair_2_4",
        )
        report = replace_linear_with_cutlass_sparse_nvfp4(
            model,
            "fakevlm",
            CutlassSparseNVFP4Config(prune=False),
        )
        return report_metadata(args.method, report, prune_report, "dynamic_tensor_global_scale_online")
    if args.method == "marlin_weight_only":
        report = replace_linear_with_marlin_nvfp4(
            model,
            "fakevlm",
            MarlinNVFP4Config(activation_dtype=torch.bfloat16),
        )
        return report_metadata(args.method, report, None, "none_bf16_activation")
    if args.method == "dense_nvfp4_prefill_marlin_decode":
        report = replace_linear_with_dense_nvfp4_prefill_marlin_decode(
            model,
            decode_m_threshold=args.decode_m_threshold,
        )
        return report_metadata(args.method, report, None, "dynamic_online_for_prefill_bf16_for_decode")
    raise ValueError(f"Unsupported method: {args.method}")


def apply_calibrated_pruning(
    model: nn.Module,
    selected: list[Any],
    calib_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    pattern: str,
) -> dict[str, Any]:
    print(f"[calibration] collect hessian pattern={pattern} samples={args.calib_samples}")
    hessian = collect_vlm_hessian_diag(
        model=model,
        modules=selected,
        dataloader=calib_loader,
        device=device,
        input_dtype=torch.bfloat16,
        max_samples=args.calib_samples,
    )
    records = []
    skipped = []
    for info in selected:
        matrix = flatten_weight(info.module)
        hdiag = hessian.get(info.name)
        if pattern == "dense_2_4":
            result = prune_dense_2_4(matrix, hdiag)
        elif pattern == "nvfp4_pair_2_4":
            result = prune_nvfp4_pair_2_4(matrix, hdiag)
        else:
            raise ValueError(f"Unsupported prune pattern: {pattern}")
        if result.mask is None:
            skipped.append({"name": info.name, **result.stats})
        else:
            info.module.weight.data.copy_(restore_weight_shape(info.module, result.weight))
        records.append({"name": info.name, **result.stats})
    return {
        "calibration_used": True,
        "calib_samples": args.calib_samples,
        "calib_batch_size": args.calib_batch_size,
        "pattern": pattern,
        "modules": records,
        "skipped": skipped,
    }


@torch.inference_mode()
def collect_vlm_hessian_diag(
    *,
    model: nn.Module,
    modules: list[Any],
    dataloader: DataLoader,
    device: torch.device,
    input_dtype: torch.dtype,
    max_samples: int,
) -> dict[str, torch.Tensor]:
    stats = {info.name: torch.zeros(info.columns, dtype=torch.float64) for info in modules}
    counts = {info.name: 0 for info in modules}
    handles = []

    def make_hook(info):
        def hook(module: nn.Module, inputs, output) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            x = inputs[0].reshape(-1, inputs[0].shape[-1])
            stats[info.name] += x.detach().float().pow(2).sum(dim=0).double().cpu()
            counts[info.name] += int(x.shape[0])

        return hook

    for info in modules:
        handles.append(info.module.register_forward_hook(make_hook(info)))

    processed = 0
    try:
        for batch in tqdm(dataloader, desc="calib", leave=False):
            remaining = max_samples - processed
            if remaining <= 0:
                break
            inputs = move_inputs(batch["inputs"], device, input_dtype)
            current = int(next(iter(inputs.values())).shape[0])
            if current > remaining:
                inputs = {key: value[:remaining] for key, value in inputs.items()}
                current = remaining
            model(**inputs, use_cache=False)
            processed += current
    finally:
        for handle in handles:
            handle.remove()

    output = {}
    for name, total in stats.items():
        count = counts[name]
        output[name] = (total / count).float() if count else torch.ones_like(total, dtype=torch.float32)
    return output


def replace_linear_with_dense_nvfp4_prefill_marlin_decode(
    model: nn.Module,
    *,
    decode_m_threshold: int,
) -> dict[str, Any]:
    wrapper = _load_wrapper()
    selected = select_compressible_modules(model, "fakevlm")
    skipped: list[dict[str, str]] = []
    replaced = 0
    for info in selected:
        if info.kind != "linear":
            skipped.append({"name": info.name, "reason": f"unsupported_kind:{info.kind}"})
            continue
        parent, child_name = resolve_parent(model, info.name)
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": info.name, "reason": f"not_linear:{type(linear).__name__}"})
            continue
        if not wrapper.can_use_cutlass_nvfp4(1, linear.out_features, linear.in_features, load_extension=False):
            skipped.append({"name": info.name, "reason": "shape_not_supported:dense_nvfp4"})
            continue
        if not wrapper.can_use_marlin_nvfp4(1, linear.out_features, linear.in_features, load_extension=False):
            skipped.append({"name": info.name, "reason": "shape_not_supported:marlin_nvfp4"})
            continue
        canonical = wrapper.canonical_from_linear(linear, device=linear.weight.device)
        setattr(
            parent,
            child_name,
            QwenHybridDenseNVFP4Linear(
                canonical,
                decode_activation_dtype=torch.bfloat16,
                marlin_m_threshold=decode_m_threshold,
                prefill_backend="dense_nvfp4",
                decode_backend="marlin_nvfp4",
            ),
        )
        replaced += 1
    return {
        "backend": "dense_nvfp4_prefill_marlin_decode",
        "config": {
            "decode_activation_dtype": "torch.bfloat16",
            "decode_m_threshold": decode_m_threshold,
            "prefill_backend": "dense_nvfp4",
            "decode_backend": "marlin_nvfp4",
        },
        "replaced_linear_count": replaced,
        "skipped_linear_count": len(skipped),
        "skipped": skipped,
    }


@torch.inference_mode()
def validate(
    args: argparse.Namespace,
    model: nn.Module,
    processor: Any,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    for batch in tqdm(dataloader, desc=args.method):
        inputs = move_inputs(batch["inputs"], device, torch.bfloat16)
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        labels = batch["label"].tolist()
        paths = list(batch["image_path"])
        cates = list(batch["cate"])
        for i in range(output.shape[0]):
            response = processor.decode(output[i], skip_special_tokens=True).split("?")[-1]
            pred = classify_response(response)
            label = int(labels[i])
            cate = str(cates[i])
            predictions.append(
                {
                    "image_path": paths[i],
                    "label": label,
                    "pred": pred,
                    "output": response,
                }
            )
            update_result_counts(results, cate, label, pred)
    return {"predictions": predictions, "accuracy": calculate_results_acc(results)}


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device, input_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=input_dtype, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


def classify_response(response: str) -> int:
    first = response.split(".")[0].lower()
    if "real" in first:
        return 1
    if "fake" in first:
        return 0
    parts = response.split(".")
    if len(parts) > 1:
        second = parts[1].lower()
        if "real" in second:
            return 1
        if "fake" in second:
            return 0
    return random.choice([0, 1])


def update_result_counts(results: dict[str, Any], cate: str, label: int, pred: int) -> None:
    if cate not in results:
        results[cate] = {"right": {"right_fake": 0, "right_real": 0}, "wrong": {"wrong_fake": 0, "wrong_real": 0}}
    if label == pred:
        if label == 1:
            results[cate]["right"]["right_real"] += 1
        else:
            results[cate]["right"]["right_fake"] += 1
    else:
        if label == 1:
            results[cate]["wrong"]["wrong_real"] += 1
        else:
            results[cate]["wrong"]["wrong_fake"] += 1


def calculate_results_acc(results: dict[str, Any]) -> dict[str, Any]:
    acc_results = {}
    for cate, data in results.items():
        right_real = data["right"]["right_real"]
        right_fake = data["right"]["right_fake"]
        wrong_real = data["wrong"]["wrong_real"]
        wrong_fake = data["wrong"]["wrong_fake"]
        total_real = right_real + wrong_real
        total_fake = right_fake + wrong_fake
        total = total_real + total_fake
        acc_results[cate] = {
            "total_samples": total,
            "total_accuracy": round((right_real + right_fake) / total, 4) if total else 0,
            "real_accuracy": round(right_real / total_real, 4) if total_real else 0,
            "fake_accuracy": round(right_fake / total_fake, 4) if total_fake else 0,
            "confusion_matrix": {
                "right_real": right_real,
                "wrong_real": wrong_real,
                "right_fake": right_fake,
                "wrong_fake": wrong_fake,
            },
        }
    total_right = sum(r["right"]["right_real"] + r["right"]["right_fake"] for r in results.values())
    total_wrong = sum(r["wrong"]["wrong_real"] + r["wrong"]["wrong_fake"] for r in results.values())
    return {
        "category_acc": acc_results,
        "global_stats": {
            "total_right": total_right,
            "total_wrong": total_wrong,
            "global_accuracy": total_right / (total_right + total_wrong) if (total_right + total_wrong) else 0,
        },
    }


def report_metadata(method: str, report: Any, prune_report: dict[str, Any] | None, activation_quant: str) -> dict[str, Any]:
    if is_dataclass(report):
        payload = asdict(report)
    else:
        payload = dict(report)
    return {
        "method": method,
        "replacement_backend": payload.get("backend", ""),
        "replacement_config": payload.get("config", {}),
        "replaced_linear_count": int(payload.get("replaced_linear_count", 0)),
        "skipped_linear_count": int(payload.get("skipped_linear_count", 0)),
        "skipped": payload.get("skipped", []),
        "calibration_used": prune_report is not None,
        "prune_report": prune_report,
        "activation_quant": activation_quant,
    }


def resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": args.method,
        "model_path": args.model_path,
        "test_json_file": args.test_json_file,
        "image_root": args.image_root,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "sample_limit": args.sample_limit,
        "calib_samples": args.calib_samples,
        "calib_batch_size": args.calib_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decode_m_threshold": args.decode_m_threshold,
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def write_summary_csv(path: Path, method: str, accuracy: dict[str, Any], prep: dict[str, Any]) -> None:
    row = {
        "method": method,
        "total_right": accuracy["global_stats"]["total_right"],
        "total_wrong": accuracy["global_stats"]["total_wrong"],
        "global_accuracy": f"{accuracy['global_stats']['global_accuracy']:.6f}",
        "replaced_linear_count": prep.get("replaced_linear_count", ""),
        "skipped_linear_count": prep.get("skipped_linear_count", ""),
        "replacement_backend": prep.get("replacement_backend", ""),
        "activation_quant": prep.get("activation_quant", ""),
        "calibration_used": prep.get("calibration_used", ""),
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
