from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fake.compression.int4 import (
    INT4Config,
    calculate_int4_group_scales,
    int4_scale_stats,
    quantize_int4_column_grouped,
)
from fake.compression.modules import ModuleInfo, flatten_weight


@dataclass
class SparseGPTResult:
    weight: torch.Tensor
    mask: torch.Tensor | None
    scales: torch.Tensor | None
    prune_stats: dict[str, object]
    quant_stats: dict[str, object]


@torch.inference_mode()
def sparsegpt_int4_compress_module(
    model: nn.Module,
    info: ModuleInfo,
    dataloader: DataLoader,
    device: str | torch.device,
    input_dtype: torch.dtype,
    max_samples: int,
    method: str,
    sparsity: float,
    int4_config: INT4Config,
    block_size: int,
    percdamp: float,
) -> SparseGPTResult:
    matrix = flatten_weight(info.module).detach()
    if matrix.shape[-1] % int4_config.group_size != 0:
        stats = {
            "status": "skipped",
            "reason": "columns_not_divisible_by_int4_group_size",
            "columns": matrix.shape[-1],
            "group_size": int4_config.group_size,
        }
        return SparseGPTResult(matrix, None, None, stats, stats)
    if method == "int4_semi_structured_sparse" and matrix.shape[-1] % 8 != 0:
        stats = {"status": "skipped", "reason": "columns_not_divisible_by_8", "columns": matrix.shape[-1]}
        return SparseGPTResult(matrix, None, None, stats, stats)

    hessian = collect_module_hessian(
        model=model,
        info=info,
        dataloader=dataloader,
        device=device,
        input_dtype=input_dtype,
        max_samples=max_samples,
    )
    if not torch.isfinite(hessian).all():
        raise RuntimeError(f"SparseGPT Hessian contains non-finite values for module {info.name}")

    scales = calculate_int4_group_scales(matrix, int4_config)
    result_weight, mask = _sparsegpt_prune_and_quantize(
        matrix=matrix,
        hessian=hessian.to(device=matrix.device),
        method=method,
        sparsity=sparsity,
        int4_config=int4_config,
        scales=scales.to(device=matrix.device),
        block_size=block_size,
        percdamp=percdamp,
        module_name=info.name,
    )
    prune_stats = _mask_stats(mask, method)
    prune_stats.update({"sparsegpt_block_size": block_size, "sparsegpt_percdamp": percdamp})
    quant_stats = int4_scale_stats(scales, int4_config)
    quant_stats.update({"sparsegpt": True})
    return SparseGPTResult(result_weight, mask, scales.cpu(), prune_stats, quant_stats)


@torch.inference_mode()
def collect_module_hessian(
    model: nn.Module,
    info: ModuleInfo,
    dataloader: DataLoader,
    device: str | torch.device,
    input_dtype: torch.dtype,
    max_samples: int,
) -> torch.Tensor:
    columns = info.columns
    hessian = torch.zeros((columns, columns), device=device, dtype=torch.float32)
    samples = 0

    def hook(module: nn.Module, inputs, output) -> None:
        nonlocal hessian, samples
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        flat = _flatten_input(module, x)
        if flat is None or flat.numel() == 0:
            return
        flat = flat.detach().float()
        batch = flat.shape[0]
        hessian *= samples / (samples + batch)
        samples += batch
        flat = (2.0 / samples) ** 0.5 * flat
        hessian += flat.t().matmul(flat)

    handle = info.module.register_forward_hook(hook)
    processed = 0
    try:
        for images, _ in dataloader:
            remaining = max_samples - processed
            if remaining <= 0:
                break
            if images.shape[0] > remaining:
                images = images[:remaining]
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            model(images)
            processed += images.shape[0]
    finally:
        handle.remove()

    if samples == 0:
        raise RuntimeError(f"No calibration samples collected for SparseGPT module {info.name}")
    return hessian


def _sparsegpt_prune_and_quantize(
    matrix: torch.Tensor,
    hessian: torch.Tensor,
    method: str,
    sparsity: float,
    int4_config: INT4Config,
    scales: torch.Tensor,
    block_size: int,
    percdamp: float,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError(f"sparsegpt_block_size must be positive, got {block_size}")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if method == "int4_semi_structured_sparse" and block_size % 8 != 0:
        raise ValueError(f"sparsegpt_block_size must be divisible by 8 for {method}, got {block_size}")

    original = matrix.detach().float()
    W = original.clone()
    H = hessian.float()
    diag = torch.diag(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1
        W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    indices = torch.arange(H.shape[0], device=H.device)
    H[indices, indices] += damp
    try:
        chol = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(chol)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except RuntimeError as exc:
        raise RuntimeError(f"SparseGPT Cholesky failed for module {module_name}") from exc
    if not torch.isfinite(Hinv).all():
        raise RuntimeError(f"SparseGPT H inverse contains non-finite values for module {module_name}")

    full_mask = torch.ones_like(W, dtype=torch.bool)
    for i1 in range(0, W.shape[1], block_size):
        i2 = min(i1 + block_size, W.shape[1])
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        mask1 = _block_prune_mask(W1, Hinv1, method, sparsity)

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()
            q[~mask1[:, i]] = 0
            q = quantize_int4_column_grouped(q, scales, i1 + i, int4_config)
            q[~mask1[:, i]] = 0

            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        full_mask[:, i1:i2] = mask1

    return W.reshape_as(matrix).to(matrix.dtype), full_mask


def _block_prune_mask(W1: torch.Tensor, Hinv1: torch.Tensor, method: str, sparsity: float) -> torch.Tensor:
    score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2).clamp(min=1e-12)
    if method == "int4_unstructured_sparse":
        num_prune = int(score.numel() * sparsity)
        mask = torch.ones_like(score, dtype=torch.bool)
        if num_prune > 0:
            prune_idx = torch.topk(score.reshape(-1), k=num_prune, largest=False).indices
            flat_mask = mask.reshape(-1)
            flat_mask[prune_idx] = False
        return mask
    if method == "int4_semi_structured_sparse":
        if score.shape[1] % 8 != 0:
            raise RuntimeError("SparseGPT semi-structured block columns must be divisible by 8")
        pair_score = score.reshape(score.shape[0], -1, 4, 2).sum(dim=-1)
        keep_pairs = torch.ones_like(pair_score, dtype=torch.bool)
        prune_pair_idx = torch.topk(pair_score, k=2, dim=-1, largest=False).indices
        keep_pairs.scatter_(-1, prune_pair_idx, False)
        return keep_pairs.unsqueeze(-1).expand(score.shape[0], score.shape[1] // 8, 4, 2).reshape_as(score)
    raise ValueError(f"Unsupported SparseGPT INT4 method: {method}")


def _flatten_input(module: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    if isinstance(module, nn.Linear):
        return x.reshape(-1, x.shape[-1])
    if isinstance(module, nn.Conv2d):
        unfolded = F.unfold(
            x,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        return unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
    return None


def _mask_stats(mask: torch.Tensor, method: str) -> dict[str, object]:
    zeros = int((~mask).sum().item())
    stats: dict[str, object] = {
        "status": "ok",
        "numel": int(mask.numel()),
        "zeros": zeros,
        "actual_sparsity": zeros / mask.numel() if mask.numel() else 0.0,
        "algorithm": "sparsegpt_full_hessian",
    }
    if method == "int4_unstructured_sparse":
        stats["pattern"] = "unstructured"
    else:
        stats["pattern"] = "int4_pair_2_4_over_8"
    return stats
