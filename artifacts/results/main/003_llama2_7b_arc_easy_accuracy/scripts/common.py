#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[5]
CUTLASS_WRAPPER_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
for path in (REPO_ROOT, CUTLASS_WRAPPER_ROOT, CUTLASS_WRAPPER_ROOT / "modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("HF_HOME", "/home/agent/wja/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/agent/wja/.cache/huggingface/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/agent/wja/.cache/huggingface/transformers")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "llama2-7b": {
        "label": "Llama-2-7B",
        "path": "/home/agent/wja/data/models/LLM-Research/llama-2-7b",
        "family": "llama",
        "loader": "causal_lm",
        "trust_remote_code": False,
    },
    "llama31-8b": {
        "label": "Llama-3.1-8B-Instruct",
        "path": "/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "family": "llama",
        "loader": "causal_lm",
        "trust_remote_code": False,
    },
    "qwen35-9b": {
        "label": "Qwen3.5-9B",
        "path": "/home/agent/wja/data/models/Qwen/Qwen3.5-9B",
        "family": "qwen3_5",
        "loader": "causal_lm",
        "trust_remote_code": True,
    },
}
DEFAULT_MODEL_KEY = "llama2-7b"
METHODS = (
    "dense_bf16",
    "sparse_bf16",
    "dense_nvfp4",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)
COMPRESSED_METHODS = (
    "sparse_bf16",
    "dense_nvfp4",
    "sparse_nvfp4",
    "marlin_nvfp4",
)


@dataclass(frozen=True)
class CalibConfig:
    samples: int = 128
    seq_len: int = 2048
    seed: int = 0
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    cache_dir: str = "/home/agent/wja/.cache/huggingface"


@dataclass(frozen=True)
class CompressionConfig:
    method: str
    sparsity: float = 0.5
    sparsegpt_block_size: int = 128
    sparsegpt_percdamp: float = 0.01
    nvfp4_group_size: int = 16
    nvfp4_scale_precision: str = "fp16"
    nvfp4_scale_rule: str = "four_over_six_mse"
    nvfp4_scale_remap: str = "none"


class TokenBlockDataset(Dataset[torch.Tensor]):
    def __init__(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected 2D token blocks, got shape={tuple(input_ids.shape)}")
        self.input_ids = input_ids

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.input_ids[idx]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_spec(model_key: str) -> dict[str, Any]:
    if model_key not in MODELS:
        raise ValueError(f"Unsupported model: {model_key}")
    return MODELS[model_key]


def model_result_root(output_root: Path, model_key: str) -> Path:
    if model_key == DEFAULT_MODEL_KEY:
        return output_root
    return output_root / "models" / model_key


def load_model(model_key: str, *, device: str, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoModel, AutoModelForCausalLM

    spec = model_spec(model_key)
    cls = AutoModelForCausalLM if spec["loader"] == "causal_lm" else AutoModel
    model = cls.from_pretrained(
        spec["path"],
        torch_dtype=dtype,
        trust_remote_code=bool(spec["trust_remote_code"]),
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def load_tokenizer(model_key: str):
    from transformers import AutoTokenizer

    spec = model_spec(model_key)
    return AutoTokenizer.from_pretrained(
        spec["path"],
        local_files_only=True,
        use_fast=True,
        trust_remote_code=bool(spec["trust_remote_code"]),
    )


def build_wikitext2_blocks(config: CalibConfig, *, model_key: str) -> tuple[torch.Tensor, dict[str, Any]]:
    from datasets import load_dataset

    tokenizer = load_tokenizer(model_key)
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
        cache_dir=config.cache_dir,
    )
    texts = [row["text"] for row in dataset if row.get("text") and row["text"].strip()]
    if not texts:
        raise RuntimeError("WikiText-2 returned no non-empty calibration text")
    joined = "\n\n".join(texts)
    tokenized = tokenizer(joined, return_tensors="pt", add_special_tokens=False).input_ids[0]
    total_needed = config.samples * config.seq_len
    if int(tokenized.numel()) < total_needed:
        raise RuntimeError(
            f"Not enough WikiText-2 tokens for calibration: have={tokenized.numel()} need={total_needed}"
        )
    generator = random.Random(config.seed)
    max_start = int(tokenized.numel()) - config.seq_len
    starts = [generator.randint(0, max_start) for _ in range(config.samples)]
    blocks = torch.stack([tokenized[start : start + config.seq_len] for start in starts]).long()
    metadata = {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "cache_dir": config.cache_dir,
        "samples": config.samples,
        "seq_len": config.seq_len,
        "seed": config.seed,
        "token_count": int(tokenized.numel()),
        "sample_starts": starts,
        "timestamp": utc_now(),
    }
    return blocks, metadata


def build_calib_loader(blocks: torch.Tensor, *, batch_size: int) -> DataLoader:
    return DataLoader(TokenBlockDataset(blocks), batch_size=batch_size, shuffle=False)


@torch.inference_mode()
def run_lm_batch(model: nn.Module, batch: torch.Tensor, *, device: str) -> None:
    input_ids = batch.to(device=device, non_blocking=True)
    model(input_ids=input_ids, use_cache=False)


def compressible_modules(model: nn.Module, model_key: str):
    from fake.compression.modules import select_compressible_modules

    return [info for info in select_compressible_modules(model, model_spec(model_key)["family"]) if info.kind == "linear"]


def module_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def flatten_linear_input(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


@torch.inference_mode()
def collect_hessian_diag_llama(
    model: nn.Module,
    modules: list[Any],
    loader: DataLoader,
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    stats = {info.name: torch.zeros(info.columns, dtype=torch.float64) for info in modules}
    counts = {info.name: 0 for info in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs, _output) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            flat = flatten_linear_input(inputs[0]).detach().float()
            stats[name] += flat.pow(2).sum(dim=0).double().cpu()
            counts[name] += int(flat.shape[0])

        return hook

    for info in modules:
        handles.append(info.module.register_forward_hook(make_hook(info.name)))
    try:
        for batch in loader:
            run_lm_batch(model, batch, device=device)
    finally:
        for handle in handles:
            handle.remove()

    out: dict[str, torch.Tensor] = {}
    for name, values in stats.items():
        count = counts[name]
        out[name] = (values / max(count, 1)).float()
    return out


@torch.inference_mode()
def collect_module_hessian_llama(
    model: nn.Module,
    module: nn.Module,
    loader: DataLoader,
    *,
    columns: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    hessian = torch.zeros((columns, columns), device=device, dtype=torch.float32)
    samples = 0

    def hook(_module: nn.Module, inputs, _output) -> None:
        nonlocal hessian, samples
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        flat = flatten_linear_input(inputs[0]).detach().float()
        batch = int(flat.shape[0])
        if batch == 0:
            return
        hessian *= samples / (samples + batch)
        samples += batch
        scaled = (2.0 / samples) ** 0.5 * flat
        hessian += scaled.t().matmul(scaled)

    handle = module.register_forward_hook(hook)
    try:
        for batch in loader:
            run_lm_batch(model, batch, device=device)
    finally:
        handle.remove()
    if samples == 0:
        raise RuntimeError("No calibration samples reached target module")
    return hessian, samples


@torch.inference_mode()
def collect_hessians_for_modules_llama(
    model: nn.Module,
    modules: list[Any],
    loader: DataLoader,
    *,
    device: str,
) -> dict[str, tuple[torch.Tensor, int]]:
    hessians = {
        info.name: torch.zeros((info.columns, info.columns), device=device, dtype=torch.float32)
        for info in modules
    }
    samples = {info.name: 0 for info in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs, _output) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            flat = flatten_linear_input(inputs[0]).detach().float()
            batch = int(flat.shape[0])
            if batch == 0:
                return
            hessians[name] *= samples[name] / (samples[name] + batch)
            samples[name] += batch
            scaled = (2.0 / samples[name]) ** 0.5 * flat
            hessians[name] += scaled.t().matmul(scaled)

        return hook

    for info in modules:
        handles.append(info.module.register_forward_hook(make_hook(info.name)))
    try:
        for batch in loader:
            run_lm_batch(model, batch, device=device)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name, count in samples.items() if count == 0]
    if missing:
        raise RuntimeError(f"No calibration samples reached modules: {missing[:5]}")
    return {name: (hessians[name], samples[name]) for name in hessians}


def sparsegpt_prune_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    pattern: str,
    sparsity: float,
    block_size: int,
    percdamp: float,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if pattern not in {"dense_2_4", "nvfp4_pair_2_4_over_8"}:
        raise ValueError(f"Unsupported sparse pattern: {pattern}")
    if pattern == "dense_2_4" and weight.shape[-1] % 4 != 0:
        raise ValueError(f"{module_name}: columns not divisible by 4")
    if pattern == "nvfp4_pair_2_4_over_8" and weight.shape[-1] % 8 != 0:
        raise ValueError(f"{module_name}: columns not divisible by 8")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0,1), got {sparsity}")

    original_dtype = weight.dtype
    W = weight.detach().float().clone()
    H = hessian.float().clone()
    diag = torch.diag(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1
        W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(H.shape[0], device=H.device)
    H[idx, idx] += damp
    try:
        chol = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(chol)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except RuntimeError as exc:
        raise RuntimeError(f"SparseGPT Cholesky failed for {module_name}") from exc
    if not torch.isfinite(Hinv).all():
        raise RuntimeError(f"SparseGPT H inverse contains non-finite values for {module_name}")

    full_mask = torch.ones_like(W, dtype=torch.bool)
    for i1 in range(0, W.shape[1], block_size):
        i2 = min(i1 + block_size, W.shape[1])
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        mask1 = sparsegpt_block_mask(W1, Hinv1, pattern=pattern, sparsity=sparsity)
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i].clamp(min=1e-12)
            q = w.clone()
            q[~mask1[:, i]] = 0
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        full_mask[:, i1:i2] = mask1

    zeros = int((~full_mask).sum().item())
    stats = {
        "status": "ok",
        "algorithm": "sparsegpt_full_hessian",
        "pattern": pattern,
        "sparsegpt_block_size": block_size,
        "sparsegpt_percdamp": percdamp,
        "numel": int(full_mask.numel()),
        "zeros": zeros,
        "actual_sparsity": zeros / full_mask.numel() if full_mask.numel() else 0.0,
    }
    return W.to(original_dtype), full_mask.detach().cpu(), stats


def sparsegpt_block_mask(
    W1: torch.Tensor,
    Hinv1: torch.Tensor,
    *,
    pattern: str,
    sparsity: float,
) -> torch.Tensor:
    score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2).clamp(min=1e-12)
    if pattern == "dense_2_4":
        score4 = score.reshape(score.shape[0], -1, 4)
        keep = torch.ones_like(score4, dtype=torch.bool)
        prune_idx = torch.topk(score4, k=2, dim=-1, largest=False).indices
        keep.scatter_(-1, prune_idx, False)
        return keep.reshape_as(score)
    if pattern == "nvfp4_pair_2_4_over_8":
        score_pair = score.reshape(score.shape[0], -1, 4, 2).sum(dim=-1)
        keep_pairs = torch.ones_like(score_pair, dtype=torch.bool)
        prune_pair_idx = torch.topk(score_pair, k=2, dim=-1, largest=False).indices
        keep_pairs.scatter_(-1, prune_pair_idx, False)
        return keep_pairs.unsqueeze(-1).expand(score.shape[0], score.shape[1] // 8, 4, 2).reshape_as(score)
    num_prune = int(score.numel() * sparsity)
    mask = torch.ones_like(score, dtype=torch.bool)
    if num_prune > 0:
        prune_idx = torch.topk(score.reshape(-1), k=num_prune, largest=False).indices
        mask.reshape(-1)[prune_idx] = False
    return mask


def nvfp4_quantize_weight(
    weight: torch.Tensor,
    config: CompressionConfig,
    hessian_diag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    from fake.compression.nvfp4 import FP4_E2M1_MAX, NVFP4Config, cast_to_fp4, fake_quantize_nvfp4_weight

    if hessian_diag is None or config.nvfp4_scale_rule != "four_over_six_mse":
        result = fake_quantize_nvfp4_weight(
            weight,
            NVFP4Config(
                group_size=config.nvfp4_group_size,
                scale_precision=config.nvfp4_scale_precision,
                scale_rule=config.nvfp4_scale_rule,
                scale_remap=config.nvfp4_scale_remap,
            ),
        )
        return result.weight, result.scales, result.stats
    if config.nvfp4_group_size not in (16, 32):
        return weight, None, {"status": "skipped", "reason": "invalid_group_size"}
    if weight.shape[-1] % config.nvfp4_group_size != 0:
        return (
            weight,
            None,
            {
                "status": "skipped",
                "reason": "columns_not_divisible_by_group_size",
                "columns": weight.shape[-1],
                "group_size": config.nvfp4_group_size,
            },
        )
    original_dtype = weight.dtype
    x = weight.detach().float()
    grouped = x.reshape(x.shape[0], -1, config.nvfp4_group_size)
    h = hessian_diag.to(device=x.device, dtype=torch.float32).clamp(min=1e-12)
    h_grouped = h.reshape(1, -1, config.nvfp4_group_size)

    scales_6 = (grouped.abs().amax(dim=-1, keepdim=True) / FP4_E2M1_MAX).clamp(min=1e-12)
    dequant_6 = cast_to_fp4(grouped / scales_6) * scales_6
    err_6 = ((dequant_6 - grouped).pow(2) * h_grouped).mean(dim=-1, keepdim=True)

    scales_4 = (grouped.abs().amax(dim=-1, keepdim=True) / 4.0).clamp(min=1e-12)
    dequant_4 = cast_to_fp4(grouped / scales_4) * scales_4
    err_4 = ((dequant_4 - grouped).pow(2) * h_grouped).mean(dim=-1, keepdim=True)

    use_4 = err_4 < err_6
    dequant = torch.where(use_4, dequant_4, dequant_6).reshape_as(x).to(original_dtype)
    scales = torch.where(use_4, scales_4, scales_6).squeeze(-1).to(torch.float16).cpu()
    groups_4 = int(use_4.sum().item())
    groups_total = int(use_4.numel())
    stats = {
        "status": "ok",
        "group_size": config.nvfp4_group_size,
        "scale_precision": config.nvfp4_scale_precision,
        "scale_rule": "four_over_six_mse",
        "scale_remap": config.nvfp4_scale_remap,
        "calibration": "hessian_diag_weighted_wikitext2",
        "num_groups": int(scales.numel()),
        "scale_min": float(scales.float().min().item()),
        "scale_max": float(scales.float().max().item()),
        "scale_denominator_6_groups": groups_total - groups_4,
        "scale_denominator_4_groups": groups_4,
    }
    return dequant, scales, stats


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {value}")


def replacement_report_dict(report: Any) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    out: dict[str, Any] = {}
    for key in ("backend", "config", "replaced_linear_count", "skipped_linear_count", "skipped"):
        if hasattr(report, key):
            out[key] = getattr(report, key)
    if hasattr(report, "csv_fields"):
        out["csv_fields"] = report.csv_fields()
    return out
