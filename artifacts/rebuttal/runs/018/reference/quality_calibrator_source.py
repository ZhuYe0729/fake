from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from daso.policy import METHODS, PHASES


FUSED_TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def fused_type(module_name: str) -> str:
    suffix = module_name.rsplit(".", 1)[-1]
    if suffix in {"q_proj", "k_proj", "v_proj"}:
        return "qkv_proj"
    if suffix in {"gate_proj", "up_proj"}:
        return "gate_up_proj"
    if suffix not in FUSED_TYPES:
        raise ValueError(f"unsupported module type: {module_name}")
    return suffix


def build_policy_features(policy: dict[str, Any], local_errors: dict[tuple[str, int, str, str], float],
                          *, buckets: int = 4, layers_per_bucket: int = 8) -> torch.Tensor:
    value = torch.zeros((len(PHASES), len(METHODS), buckets, len(FUSED_TYPES)), dtype=torch.float64)
    for name, actions in policy["method_map"].items():
        bucket = int(name.split(".")[2]) // layers_per_bucket
        kind = fused_type(name)
        for phase_index, phase in enumerate(PHASES):
            method = actions[f"{phase}_method"]
            value[phase_index, METHODS.index(method), bucket, FUSED_TYPES.index(kind)] += \
                local_errors[phase, bucket, kind, method]
    return value


@dataclass
class QualityFit:
    feature_scale: float
    bias: float
    coefficients: torch.Tensor
    raw_parameters: dict[str, list[float] | float]

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.bias + (features / self.feature_scale * self.coefficients).sum((-4, -3, -2, -1))

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_scale": self.feature_scale, "bias": self.bias,
            "coefficients": self.coefficients.tolist(), **self.raw_parameters,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")


class QualityCalibrator:
    """Positive local-error calibration used by the final phase-aware model."""

    def __init__(self, *, steps: int = 4000, learning_rate: float = 0.03,
                 regularization: float = 0.05):
        self.steps = steps
        self.learning_rate = learning_rate
        self.regularization = regularization

    def fit(self, features: torch.Tensor, labels: torch.Tensor,
            train_mask: torch.Tensor) -> QualityFit:
        x = features.to(torch.float64)
        y = labels.to(torch.float64)
        train = train_mask.to(torch.bool)
        scale = x[train].sum((1, 2, 3, 4)).mean().clamp(min=1e-12)
        x = x / scale
        global_factor = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
        phase = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        method = torch.zeros(len(METHODS), dtype=torch.float64, requires_grad=True)
        bucket = torch.zeros(x.shape[3], dtype=torch.float64, requires_grad=True)
        kind = torch.zeros(x.shape[4], dtype=torch.float64, requires_grad=True)
        bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        params = [global_factor, phase, method, bucket, kind, bias]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        for _ in range(self.steps):
            optimizer.zero_grad()
            coefficients = torch.relu(global_factor + phase[:, None, None, None]
                + method[None, :, None, None] + bucket[None, None, :, None]
                + kind[None, None, None, :])
            prediction = bias + (x * coefficients).sum((1, 2, 3, 4))
            loss = ((prediction[train] - y[train]) ** 2).mean()
            loss += self.regularization * sum((item * item).mean() for item in params[:-1])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            coefficients = torch.relu(global_factor + phase[:, None, None, None]
                + method[None, :, None, None] + bucket[None, None, :, None]
                + kind[None, None, None, :]).detach()
        return QualityFit(float(scale), float(bias.detach()), coefficients, {
            "global": float(global_factor.detach()), "phase": phase.detach().tolist(),
            "method": method.detach().tolist(), "bucket": bucket.detach().tolist(),
            "type": kind.detach().tolist(),
        })

