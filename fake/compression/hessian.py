from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fake.compression.modules import ModuleInfo


@dataclass
class HessianDiagStat:
    sum_sq: torch.Tensor
    count: int = 0

    def update(self, values: torch.Tensor, count: int) -> None:
        self.sum_sq += values.detach().cpu()
        self.count += count

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return torch.ones_like(self.sum_sq)
        return self.sum_sq / self.count


@torch.inference_mode()
def collect_hessian_diag(
    model: nn.Module,
    modules: list[ModuleInfo],
    dataloader: DataLoader,
    device: str | torch.device,
    input_dtype: torch.dtype,
    max_samples: int,
) -> dict[str, torch.Tensor]:
    stats = {
        info.name: HessianDiagStat(sum_sq=torch.zeros(info.columns, dtype=torch.float64))
        for info in modules
    }
    handles = []

    def make_hook(info: ModuleInfo):
        def hook(module: nn.Module, inputs, output) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            flat, count = _flatten_module_input(module, x)
            if flat is None or count == 0:
                return
            sum_sq = flat.detach().float().pow(2).sum(dim=0).double()
            stats[info.name].update(sum_sq, count)

        return hook

    for info in modules:
        handles.append(info.module.register_forward_hook(make_hook(info)))

    processed = 0
    try:
        for batch in dataloader:
            if batch is None:
                continue
            images, _ = batch
            remaining = max_samples - processed
            if remaining <= 0:
                break
            if images.shape[0] > remaining:
                images = images[:remaining]
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            model(images)
            processed += images.shape[0]
    finally:
        for handle in handles:
            handle.remove()

    return {name: stat.mean().float() for name, stat in stats.items()}


def _flatten_module_input(module: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    if isinstance(module, nn.Linear):
        flat = x.reshape(-1, x.shape[-1])
        return flat, flat.shape[0]
    if isinstance(module, nn.Conv2d):
        unfolded = F.unfold(
            x,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        flat = unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
        return flat, flat.shape[0]
    return None, 0
