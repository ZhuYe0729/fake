from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpeedResult:
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_min_ms: float
    latency_max_ms: float
    images_per_sec: float


@torch.inference_mode()
def benchmark_forward(
    model: torch.nn.Module,
    batch_size: int,
    input_size: tuple[int, int, int],
    input_dtype: torch.dtype,
    device: str | torch.device,
    warmup: int,
    iters: int,
) -> SpeedResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for speed benchmark")

    c, h, w = input_size
    inputs = torch.randn((batch_size, c, h, w), device=device, dtype=input_dtype)
    for _ in range(warmup):
        model(inputs)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_event.record()
        model(inputs)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    mean_ms = statistics.fmean(times_ms)
    return SpeedResult(
        latency_mean_ms=mean_ms,
        latency_p50_ms=statistics.median(times_ms),
        latency_p90_ms=_percentile(times_ms, 90),
        latency_min_ms=min(times_ms),
        latency_max_ms=max(times_ms),
        images_per_sec=batch_size * 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * percentile / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight

