from __future__ import annotations

from typing import Any

import torch


def structured_mask(weight: torch.Tensor, hinv: torch.Tensor, pattern: str) -> torch.Tensor:
    score = weight.pow(2) / torch.diag(hinv).reshape(1, -1).pow(2).clamp(min=1e-12)
    if pattern == "dense_2_4":
        grouped = score.reshape(score.shape[0], -1, 4)
        keep = torch.ones_like(grouped, dtype=torch.bool)
        keep.scatter_(-1, torch.topk(grouped, k=2, dim=-1, largest=False).indices, False)
        return keep.reshape_as(score)
    if pattern == "nvfp4_pair_2_4_over_8":
        paired = score.reshape(score.shape[0], -1, 4, 2).sum(-1)
        keep = torch.ones_like(paired, dtype=torch.bool)
        keep.scatter_(-1, torch.topk(paired, k=2, dim=-1, largest=False).indices, False)
        return keep.unsqueeze(-1).expand(score.shape[0], score.shape[1] // 8, 4, 2).reshape_as(score)
    raise ValueError(pattern)


@torch.inference_mode()
def prune_weight(weight: torch.Tensor, hessian: torch.Tensor, *, pattern: str,
                 block_size: int = 128, percdamp: float = 0.01,
                 module_name: str = "") -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    divisor = 4 if pattern == "dense_2_4" else 8
    if weight.shape[-1] % divisor:
        raise ValueError(f"{module_name}: columns not divisible by {divisor}")
    original_dtype = weight.dtype
    matrix = weight.detach().float().clone()
    hessian = hessian.float().clone()
    dead = torch.diag(hessian) == 0
    if dead.any():
        hessian[dead, dead] = 1
        matrix[:, dead] = 0
    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    hessian[diagonal, diagonal] += percdamp * torch.diag(hessian).mean()
    try:
        chol = torch.linalg.cholesky(hessian)
        hinv = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    except RuntimeError as exc:
        raise RuntimeError(f"SparseGPT Cholesky failed for {module_name}") from exc
    full_mask = torch.ones_like(matrix, dtype=torch.bool)
    for start in range(0, matrix.shape[1], block_size):
        end = min(start + block_size, matrix.shape[1])
        current = matrix[:, start:end].clone()
        quantized = torch.zeros_like(current)
        errors = torch.zeros_like(current)
        local_hinv = hinv[start:end, start:end]
        mask = structured_mask(current, local_hinv, pattern)
        for column in range(end - start):
            value = current[:, column]
            kept = value.clone()
            kept[~mask[:, column]] = 0
            quantized[:, column] = kept
            error = (value - kept) / local_hinv[column, column].clamp(min=1e-12)
            current[:, column:] -= error.unsqueeze(1) @ local_hinv[column, column:].unsqueeze(0)
            errors[:, column] = error
        matrix[:, start:end] = quantized
        matrix[:, end:] -= errors @ hinv[start:end, end:]
        full_mask[:, start:end] = mask
    zeros = int((~full_mask).sum())
    metadata = {
        "algorithm": "sparsegpt_full_hessian", "pattern": pattern,
        "block_size": block_size, "percdamp": percdamp,
        "numel": full_mask.numel(), "zeros": zeros,
        "actual_sparsity": zeros / full_mask.numel(),
    }
    return matrix.to(original_dtype), full_mask.cpu(), metadata

