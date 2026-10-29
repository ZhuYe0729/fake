"""Signed additive corrections to an existing whole-policy quality prior."""
from dataclasses import dataclass

import torch


@dataclass
class ResidualQualityFit:
    coefficients: torch.Tensor

    def predict(self, features: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(prior, dtype=torch.float64) + torch.as_tensor(
            features, dtype=torch.float64) @ self.coefficients


def fit_residual_quality(features: torch.Tensor, labels: torch.Tensor,
                         prior: torch.Tensor, *, regularization: float) -> ResidualQualityFit:
    """Fit ridge residuals with no intercept, preserving the zero-feature anchor.

    Features must be sums of action features if the correction is to remain
    compatible with an additive one-hot solver. No positivity clipping is used.
    """
    x, y, base = [torch.as_tensor(value, dtype=torch.float64)
                  for value in [features, labels, prior]]
    if x.ndim != 2 or not x.shape[0] or y.shape != (x.shape[0],) or base.shape != y.shape:
        raise ValueError('features, labels and prior must align by policy')
    if regularization <= 0 or not all(torch.isfinite(v).all() for v in [x, y, base]):
        raise ValueError('finite calibration data and positive regularization are required')
    scale = x.square().mean(0).sqrt().clamp_min(1e-3)
    normalized = x / scale
    gram = normalized @ normalized.T
    gram.diagonal().add_(regularization * x.shape[0])
    coefficients = normalized.T @ torch.linalg.solve(gram, y - base) / scale
    return ResidualQualityFit(coefficients)
