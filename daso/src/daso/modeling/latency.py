from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KERNEL_ALIAS = {"w4a16_ours": "marlin_nvfp4"}


@dataclass(frozen=True)
class LatencyAction:
    kernel: str
    latency_ms: float | None
    supported: bool


class KernelLatencyBackend:
    """Adapter for the external calibrated Roofline CUTLASS predictor."""

    def __init__(self, wrapper_root: str | Path, model_root: str | Path | None = None):
        root = Path(wrapper_root).resolve()
        modeling = root / "modeling"
        if not (modeling / "kernel_predictor.py").is_file():
            raise FileNotFoundError(f"missing CUTLASS predictor: {modeling}")
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(modeling))
        from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor
        self.predictor = KernelLatencyPredictor(
            model_root=Path(model_root) if model_root else DEFAULT_MODEL_ROOT,
            kernels=("dense_bf16", "dense_nvfp4", "sparse_bf16",
                     "sparse_nvfp4", "marlin_nvfp4"),
        )

    def predict(self, m: int, n: int, k: int) -> dict[str, LatencyAction]:
        result: dict[str, LatencyAction] = {}
        for item in self.predictor.predict(m, n, k).candidates:
            result[item.kernel] = LatencyAction(
                kernel=item.kernel,
                latency_ms=None if item.latency_ms is None else float(item.latency_ms),
                supported=bool(item.supported),
            )
        return result


def compose_phase_latency(prefill_ms: float, decode_step_ms: float,
                          output_tokens: int) -> float:
    if output_tokens < 1:
        raise ValueError("output_tokens must be positive")
    return prefill_ms + max(0, output_tokens - 1) * decode_step_ms


def monotonic_calibration(predicted: list[float], measured: list[float]) -> list[float]:
    """Pool-adjacent-violators isotonic fit in ascending predicted order."""
    if len(predicted) != len(measured) or not predicted:
        raise ValueError("predicted and measured must have equal non-zero length")
    order = sorted(range(len(predicted)), key=predicted.__getitem__)
    blocks: list[dict[str, Any]] = []
    for index in order:
        blocks.append({"indices": [index], "sum": measured[index], "count": 1})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left["sum"] / left["count"] <= right["sum"] / right["count"]:
                break
            blocks[-2:] = [{"indices": left["indices"] + right["indices"],
                            "sum": left["sum"] + right["sum"],
                            "count": left["count"] + right["count"]}]
    fitted = [0.0] * len(predicted)
    for block in blocks:
        value = block["sum"] / block["count"]
        for index in block["indices"]:
            fitted[index] = value
    return fitted

