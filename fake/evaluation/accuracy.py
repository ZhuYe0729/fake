from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class AccuracyResult:
    num_samples: int
    top1: float
    top5: float
    elapsed_sec: float
    images_per_sec: float


@torch.inference_mode()
def evaluate_topk(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    input_dtype: torch.dtype,
    log_interval: int = 50,
) -> AccuracyResult:
    total = 0
    correct1 = 0
    correct5 = 0
    start = time.perf_counter()

    for step, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        logits = model(images)
        maxk = min(5, logits.shape[-1])
        _, pred = logits.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        batch_size = targets.numel()
        correct1 += correct[:1].reshape(-1).float().sum().item()
        correct5 += correct[:maxk].reshape(-1).float().sum().item()
        total += batch_size
        if log_interval > 0 and step % log_interval == 0:
            print(f"[accuracy] step={step} samples={total} top1={correct1 / total:.4f}")

    elapsed = time.perf_counter() - start
    return AccuracyResult(
        num_samples=total,
        top1=correct1 / total if total else 0.0,
        top5=correct5 / total if total else 0.0,
        elapsed_sec=elapsed,
        images_per_sec=total / elapsed if elapsed > 0 else 0.0,
    )

