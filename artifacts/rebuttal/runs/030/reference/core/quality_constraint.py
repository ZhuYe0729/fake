"""Forecast a validation guard from paired conditional-calibration samples."""
from dataclasses import dataclass

import torch


@dataclass
class QualityGuardForecast:
    mean: torch.Tensor
    standard_error: torch.Tensor
    guarded: torch.Tensor


def forecast_quality_guard(anchor_samples: torch.Tensor, group_samples: torch.Tensor,
                           decisions: torch.Tensor, *, validation_windows: int,
                           residual_margin: float = 0.) -> QualityGuardForecast:
    """Use paired covariance, including anchor/group error correlations.

    Samples are relative to BF16 or the fixed anchor, respectively. The
    residual margin comes from calibration policy holdouts. This predicts the
    validation statistic; it does not certify unknown policies or new data.
    """
    anchor, groups, values = [torch.as_tensor(x, dtype=torch.float64)
                              for x in [anchor_samples, group_samples, decisions]]
    if anchor.ndim != 1 or anchor.numel() < 2 or groups.ndim != 2 or groups.shape[1] != anchor.numel():
        raise ValueError("anchor and group samples must align by token window")
    if validation_windows < 2 or not torch.isfinite(torch.tensor(residual_margin)) or residual_margin < 0:
        raise ValueError("validation window count and residual margin must be valid")
    if values.ndim < 1 or values.shape[-1] != groups.shape[0] or not torch.all((values == 0) | (values == 1)):
        raise ValueError("decisions must be binary with one value per group")
    if not torch.isfinite(anchor).all() or not torch.isfinite(groups).all():
        raise ValueError("calibration samples must be finite")
    coefficients = torch.cat([anchor[:, None], groups.T], dim=1)
    average = coefficients.mean(0)
    centered = coefficients - average
    covariance = centered.T @ centered / (anchor.numel() - 1)
    features = torch.cat([torch.ones_like(values[..., :1]), values], dim=-1)
    mean = features @ average
    variance = ((features @ covariance) * features).sum(-1).clamp_min(0.)
    se = torch.sqrt(variance / validation_windows)
    return QualityGuardForecast(mean, se, mean + 2 * se + .005 + residual_margin)
