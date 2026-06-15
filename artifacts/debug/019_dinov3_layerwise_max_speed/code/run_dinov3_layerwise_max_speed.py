#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in (CODE_DIR, *CODE_DIR.parents) if (parent / "fake").is_dir() and (parent / "artifacts").is_dir()),
    CODE_DIR.parents[3],
)
CUTLASS_WRAPPER_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
KERNEL_PREDICTOR_PATHS = list(REPO_ROOT.glob("fake/**/modeling/kernel_predictor.py"))
MODELING_ROOT = KERNEL_PREDICTOR_PATHS[0].parents[1] if KERNEL_PREDICTOR_PATHS else CUTLASS_WRAPPER_ROOT
for path in (CODE_DIR, REPO_ROOT, CUTLASS_WRAPPER_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from dinov3_layerwise_policy import (  # noqa: E402
    DINOV3_LAYERWISE_POLICY_FORMAT,
    SUPPORTED_BACKENDS,
    DINOv3LayerPolicyItem,
    load_dinov3_vit7b16_layerwise_policy_classifier,
    policy_item_to_dict,
)
from fake.compression.modules import select_compressible_modules  # noqa: E402
from fake.evaluation.speed import benchmark_forward  # noqa: E402
from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config  # noqa: E402
from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config  # noqa: E402
from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config  # noqa: E402
from fake.models.dinov3 import DEFAULT_DINOV3_BACKBONE_PATH, DEFAULT_DINOV3_HEAD_PATH, model_input_dtype  # noqa: E402
from fake.utils.csv_io import append_csv_row  # noqa: E402
try:
    from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402
except ModuleNotFoundError as exc:
    searched = "\n".join(str(path) for path in (CUTLASS_WRAPPER_ROOT, MODELING_ROOT, *KERNEL_PREDICTOR_PATHS))
    raise ModuleNotFoundError(
        "Could not import modeling.kernel_predictor. "
        "Expected to find fake/kernels/cutlass/cutlass_wrapper/modeling/kernel_predictor.py. "
        f"Searched:\n{searched}"
    ) from exc


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/debug/019_dinov3_layerwise_max_speed"
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
BASELINE_FILES = {
    "dense": REPO_ROOT / "artifacts/results/dinov3_vit7b16_dense/speed.csv",
    "dense_nvfp4": REPO_ROOT / "artifacts/results/dinov3_vit7b16_cutlass_nvfp4/speed.csv",
    "sparse_bf16": REPO_ROOT / "artifacts/results/dinov3_vit7b16_cutlass_sparse_bf16/speed.csv",
    "sparse_nvfp4": REPO_ROOT / "artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/speed_storage.csv",
    "manual_hybrid": REPO_ROOT / "artifacts/results/dinov3_vit7b16_cutlass_hybrid/speed.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and benchmark DINOv3 layerwise max-speed CUTLASS policies.")
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--input-size", type=int, nargs=3, default=[3, 256, 256])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--kernels", nargs="+", choices=SUPPORTED_BACKENDS, default=list(SUPPORTED_BACKENDS))
    parser.add_argument("--generate-only", action="store_true", help="Write policies and summaries without loading benchmark models.")
    parser.add_argument("--no-prune", action="store_true", help="Require sparse weights to already satisfy sparse patterns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_root_readme(args)

    predictor = KernelLatencyPredictor(model_root=args.model_root, kernels=args.kernels)
    template_model, config = load_template_model(args)
    linears = enumerate_dinov3_linears(template_model)
    del template_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    seq_len = infer_dinov3_sequence_length(config, tuple(args.input_size))
    all_policy_rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        batch_dir = args.output_root / f"batch_{batch_size}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        policy_items, candidate_rows = build_policy(
            linears=linears,
            predictor=predictor,
            batch_size=batch_size,
            seq_len=seq_len,
            kernels=args.kernels,
        )
        policy_path = batch_dir / f"policy_bs{batch_size}.json"
        policy_csv_path = batch_dir / f"policy_bs{batch_size}.csv"
        candidate_csv_path = batch_dir / f"candidates_bs{batch_size}.csv"
        write_policy_json(policy_path, policy_items, args, config, seq_len)
        write_policy_csv(policy_csv_path, policy_items)
        write_csv(candidate_csv_path, candidate_rows)
        all_policy_rows.extend(policy_summary_rows(batch_size, seq_len, policy_items, candidate_rows))

        if not args.generate_only:
            benchmark_policy(args, batch_size, policy_path, config)

    write_csv(args.output_root / "policy_summary.csv", all_policy_rows)
    write_summary(args.output_root, args.batch_sizes)
    print(f"wrote DINOv3 layerwise max-speed outputs to {args.output_root}")


def load_template_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    from fake.models.dinov3 import load_dinov3_vit7b16_dense_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_dinov3_vit7b16_dense_classifier(
        backbone_path=args.backbone_path,
        head_path=args.head_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )


def enumerate_dinov3_linears(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for info in select_compressible_modules(model, "dinov3_vit7b16"):
        if info.kind != "linear":
            continue
        rows.append(
            {
                "name": info.name,
                "n": int(info.module.out_features),
                "k": int(info.module.in_features),
                "suffix": ".".join(info.name.split(".")[-2:]),
            }
        )
    return rows


def infer_dinov3_sequence_length(config: dict[str, Any], input_size: tuple[int, int, int]) -> int:
    _, height, width = input_size
    patch_size = config.get("patch_size", 16)
    if isinstance(patch_size, list):
        patch_h = int(patch_size[0])
        patch_w = int(patch_size[-1])
    else:
        patch_h = patch_w = int(patch_size)
    patch_tokens = (int(height) // patch_h) * (int(width) // patch_w)
    return 1 + int(config.get("num_register_tokens", 4)) + patch_tokens


def build_policy(
    linears: list[dict[str, Any]],
    predictor: KernelLatencyPredictor,
    batch_size: int,
    seq_len: int,
    kernels: list[str],
) -> tuple[list[DINOv3LayerPolicyItem], list[dict[str, Any]]]:
    m = int(batch_size) * int(seq_len)
    policy_items: list[DINOv3LayerPolicyItem] = []
    candidate_rows: list[dict[str, Any]] = []
    for linear in linears:
        candidates = []
        for kernel in kernels:
            pred_m = prediction_m_for_backend(m, kernel)
            selection = predictor.predict(pred_m, int(linear["n"]), int(linear["k"]))
            candidate = next((item for item in selection.candidates if item.kernel == kernel), None)
            if candidate is not None:
                candidates.append((candidate, pred_m))
        viable = [
            (candidate, pred_m)
            for candidate, pred_m in candidates
            if candidate.supported and candidate.latency_ms is not None
        ]
        if viable:
            best, _ = min(viable, key=lambda item: float(item[0].latency_ms))
            backend = str(best.kernel)
            latency_ms = float(best.latency_ms)
            reason = ""
        else:
            backend = "dense_bf16"
            latency_ms = None
            reason = "fallback_dense_bf16_no_supported_prediction"
        policy_items.append(
            DINOv3LayerPolicyItem(
                name=str(linear["name"]),
                backend=backend,
                n=int(linear["n"]),
                k=int(linear["k"]),
                predicted_latency_ms=latency_ms,
                reason=reason,
            )
        )
        for candidate, pred_m in candidates:
            candidate_rows.append(
                {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "m": m,
                    "prediction_m": pred_m,
                    "name": linear["name"],
                    "suffix": linear["suffix"],
                    "n": linear["n"],
                    "k": linear["k"],
                    "kernel": candidate.kernel,
                    "supported": candidate.supported,
                    "latency_ms": "" if candidate.latency_ms is None else candidate.latency_ms,
                    "reason": candidate.reason,
                    "source": candidate.source,
                    "prediction_status": candidate.prediction_status,
                    "selected": candidate.kernel == backend,
                }
            )
    return policy_items, candidate_rows


def prediction_m_for_backend(m: int, backend: str) -> int:
    if backend == "sparse_nvfp4":
        return round_up(m, 32)
    if backend == "sparse_bf16":
        return round_up(m, 8)
    return int(m)


def round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def benchmark_policy(args: argparse.Namespace, batch_size: int, policy_path: Path, config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --generate-only is set.")
    device = torch.device("cuda")
    model, _, report = load_dinov3_vit7b16_layerwise_policy_classifier(
        policy_path=policy_path,
        backbone_path=args.backbone_path,
        head_path=args.head_path,
        device=device,
        dense_nvfp4_config=CutlassNVFP4Config(),
        sparse_bf16_config=CutlassSparseBF16Config(prune=not args.no_prune),
        sparse_nvfp4_config=CutlassSparseNVFP4Config(prune=not args.no_prune),
    )
    input_dtype = model_input_dtype(model)
    result = benchmark_forward(
        model=model,
        batch_size=batch_size,
        input_size=tuple(args.input_size),
        input_dtype=input_dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "head": "dinov3_vit7b16_imagenet1k_linear_head",
        "method": "layerwise_max_speed_cutlass",
        "task": "forward_speed",
        "speed_scope": "random_input_classifier_forward_only",
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "input_c": args.input_size[0],
        "input_h": args.input_size[1],
        "input_w": args.input_size[2],
        "warmup": args.warmup,
        "iters": args.iters,
        "hidden_size": config.get("hidden_size", ""),
        "num_register_tokens": config.get("num_register_tokens", ""),
        "latency_mean_ms": f"{result.latency_mean_ms:.6f}",
        "latency_p50_ms": f"{result.latency_p50_ms:.6f}",
        "latency_p90_ms": f"{result.latency_p90_ms:.6f}",
        "latency_min_ms": f"{result.latency_min_ms:.6f}",
        "latency_max_ms": f"{result.latency_max_ms:.6f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "backbone_path": args.backbone_path,
        "head_path": args.head_path,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **report.csv_fields(),
    }
    append_csv_row(args.output_root / "speed.csv", list(row.keys()), row)
    print(
        "dinov3 layerwise max-speed done: "
        f"batch_size={batch_size} mean_ms={row['latency_mean_ms']} images_per_sec={row['images_per_sec']} "
        f"dense_bf16={report.dense_bf16_module_count} dense_nvfp4={report.dense_nvfp4_module_count} "
        f"sparse_bf16={report.sparse_bf16_module_count} sparse_nvfp4={report.sparse_nvfp4_module_count} "
        f"skipped={report.skipped_linear_count}"
    )
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")
    del model
    torch.cuda.empty_cache()


def write_policy_json(path: Path, items: list[DINOv3LayerPolicyItem], args: argparse.Namespace, config: dict[str, Any], seq_len: int) -> None:
    payload = {
        "policy_format": DINOV3_LAYERWISE_POLICY_FORMAT,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "input_size": list(args.input_size),
        "sequence_length": seq_len,
        "hidden_size": config.get("hidden_size"),
        "kernels": list(args.kernels),
        "modules": [policy_item_to_dict(item) for item in items],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_policy_csv(path: Path, items: list[DINOv3LayerPolicyItem]) -> None:
    write_csv(path, [policy_item_to_dict(item) for item in items])


def policy_summary_rows(
    batch_size: int,
    seq_len: int,
    items: list[DINOv3LayerPolicyItem],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(item.backend for item in items)
    predicted_total = sum(item.predicted_latency_ms or 0.0 for item in items)
    rows = [
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "summary": "selected_total",
            "backend": "layerwise",
            "module_count": len(items),
            "predicted_linear_latency_ms": predicted_total,
        }
    ]
    for backend in SUPPORTED_BACKENDS:
        rows.append(
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "summary": "selected_count",
                "backend": backend,
                "module_count": counts[backend],
                "predicted_linear_latency_ms": sum(
                    item.predicted_latency_ms or 0.0 for item in items if item.backend == backend
                ),
            }
        )
    for backend in SUPPORTED_BACKENDS:
        total = 0.0
        supported = True
        for row in candidate_rows:
            if row["kernel"] != backend:
                continue
            if not row["supported"] or row["latency_ms"] == "":
                supported = False
                break
            total += float(row["latency_ms"])
        rows.append(
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "summary": "uniform_predicted",
                "backend": backend,
                "module_count": len(items) if supported else "",
                "predicted_linear_latency_ms": total if supported else "",
            }
        )
    return rows


def write_summary(output_root: Path, batch_sizes: list[int]) -> None:
    layerwise = latest_rows_by_batch(output_root / "speed.csv")
    baseline_by_method = {method: latest_rows_by_batch(path) for method, path in BASELINE_FILES.items()}
    summary_rows = []
    for batch_size in batch_sizes:
        layer_row = layerwise.get(batch_size, {})
        best_baseline = best_baseline_for_batch(baseline_by_method, batch_size)
        layer_ips = _float(layer_row.get("images_per_sec"))
        layer_ms = _float(layer_row.get("latency_mean_ms"))
        baseline_ips = _float(best_baseline.get("images_per_sec")) if best_baseline else math.nan
        summary_rows.append(
            {
                "batch_size": batch_size,
                "layerwise_latency_mean_ms": "" if math.isnan(layer_ms) else f"{layer_ms:.6f}",
                "layerwise_images_per_sec": "" if math.isnan(layer_ips) else f"{layer_ips:.3f}",
                "best_baseline_method": best_baseline.get("_method", "") if best_baseline else "",
                "best_baseline_latency_mean_ms": "" if not best_baseline else best_baseline.get("latency_mean_ms", ""),
                "best_baseline_images_per_sec": "" if not best_baseline else best_baseline.get("images_per_sec", ""),
                "speedup_vs_best_baseline": (
                    "" if math.isnan(layer_ips) or math.isnan(baseline_ips) or baseline_ips <= 0 else f"{layer_ips / baseline_ips:.6f}"
                ),
            }
        )
    write_csv(output_root / "summary.csv", summary_rows)
    write_summary_readme(output_root, summary_rows)


def latest_rows_by_batch(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                batch = int(row.get("batch_size", ""))
            except ValueError:
                continue
            old = rows.get(batch)
            if old is None or str(row.get("timestamp", "")) >= str(old.get("timestamp", "")):
                rows[batch] = row
    return rows


def best_baseline_for_batch(baselines: dict[str, dict[int, dict[str, str]]], batch_size: int) -> dict[str, str]:
    best: dict[str, str] = {}
    best_ips = -1.0
    for method, rows in baselines.items():
        row = rows.get(batch_size)
        if not row:
            continue
        ips = _float(row.get("images_per_sec"))
        if not math.isnan(ips) and ips > best_ips:
            best = dict(row)
            best["_method"] = method
            best_ips = ips
    return best


def write_root_readme(args: argparse.Namespace) -> None:
    (args.output_root / "README.md").write_text(
        "# DINOv3 Layerwise Max Speed\n\n"
        "This directory contains speed-model-selected per-layer CUTLASS policies and real DINOv3 forward benchmarks.\n\n"
        f"- Batch sizes: `{' '.join(str(item) for item in args.batch_sizes)}`\n"
        f"- Candidate kernels: `{' '.join(args.kernels)}`\n"
        f"- Input size: `{args.input_size[0]} {args.input_size[1]} {args.input_size[2]}`\n"
        f"- Warmup/iters: `{args.warmup}/{args.iters}`\n\n"
        "Key files after a full run: `speed.csv`, `summary.csv`, `policy_summary.csv`, and per-batch policy/candidate files.\n"
    )


def write_summary_readme(output_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DINOv3 Layerwise Max Speed Summary",
        "",
        "| Batch | Layerwise img/s | Best baseline | Speedup |",
        "|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['layerwise_images_per_sec']} | "
            f"{row['best_baseline_method']} | {row['speedup_vs_best_baseline']} |"
        )
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    main()
