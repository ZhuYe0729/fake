from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PruneResult:
    weight: torch.Tensor
    mask: torch.Tensor | None
    stats: dict[str, object]


def prune_unstructured(matrix: torch.Tensor, sparsity: float, hessian_diag: torch.Tensor | None) -> PruneResult:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    score = _score(matrix, hessian_diag)
    num_prune = int(score.numel() * sparsity)
    if num_prune <= 0:
        return PruneResult(matrix, torch.ones_like(matrix, dtype=torch.bool), _stats(matrix, None, "ok"))
    flat = score.reshape(-1)
    prune_idx = torch.topk(flat, k=num_prune, largest=False).indices
    mask = torch.ones_like(flat, dtype=torch.bool)
    mask[prune_idx] = False
    mask = mask.reshape_as(matrix)
    pruned = matrix * mask.to(matrix.dtype)
    return PruneResult(pruned, mask, _stats(matrix, mask, "ok"))


def prune_dense_2_4(matrix: torch.Tensor, hessian_diag: torch.Tensor | None) -> PruneResult:
    if matrix.shape[-1] % 4 != 0:
        return PruneResult(
            matrix,
            None,
            {"status": "skipped", "reason": "columns_not_divisible_by_4", "columns": matrix.shape[-1]},
        )
    score = _score(matrix, hessian_diag).reshape(matrix.shape[0], -1, 4)
    keep = torch.ones_like(score, dtype=torch.bool)
    prune_idx = torch.topk(score, k=2, dim=-1, largest=False).indices
    keep.scatter_(-1, prune_idx, False)
    mask = keep.reshape_as(matrix)
    pruned = matrix * mask.to(matrix.dtype)
    stats = _stats(matrix, mask, "ok")
    stats["pattern"] = "dense_2_4"
    return PruneResult(pruned, mask, stats)


def prune_nvfp4_pair_2_4(matrix: torch.Tensor, hessian_diag: torch.Tensor | None) -> PruneResult:
    if matrix.shape[-1] % 8 != 0:
        return PruneResult(
            matrix,
            None,
            {"status": "skipped", "reason": "columns_not_divisible_by_8", "columns": matrix.shape[-1]},
        )
    score = _score(matrix, hessian_diag).reshape(matrix.shape[0], -1, 4, 2)
    pair_score = score.sum(dim=-1)
    keep_pairs = torch.ones_like(pair_score, dtype=torch.bool)
    prune_pair_idx = torch.topk(pair_score, k=2, dim=-1, largest=False).indices
    keep_pairs.scatter_(-1, prune_pair_idx, False)
    mask = keep_pairs.unsqueeze(-1).expand_as(score).reshape_as(matrix)
    pruned = matrix * mask.to(matrix.dtype)
    stats = _stats(matrix, mask, "ok")
    stats["pattern"] = "nvfp4_pair_2_4_over_8"
    return PruneResult(pruned, mask, stats)


def _score(matrix: torch.Tensor, hessian_diag: torch.Tensor | None) -> torch.Tensor:
    score = matrix.detach().float().pow(2)
    if hessian_diag is not None:
        h = hessian_diag.to(device=score.device, dtype=score.dtype).clamp(min=1e-12)
        score = score * h.reshape(1, -1)
    return score


def _stats(matrix: torch.Tensor, mask: torch.Tensor | None, status: str) -> dict[str, object]:
    if mask is None:
        return {"status": status, "numel": int(matrix.numel())}
    zeros = int((~mask).sum().item())
    return {
        "status": status,
        "numel": int(mask.numel()),
        "zeros": zeros,
        "actual_sparsity": zeros / mask.numel() if mask.numel() else 0.0,
    }

