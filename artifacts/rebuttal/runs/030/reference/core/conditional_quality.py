"""Paired whole-policy NLL calibration around a fixed prefill/decode anchor."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConditionalQualityFit:
    baseline_delta: float
    group_deltas: torch.Tensor
    pairs: tuple[tuple[int, int], ...]
    pair_corrections: torch.Tensor

    def predict(self, decisions: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(decisions, dtype=torch.float64)
        if values.ndim < 1 or values.shape[-1] != len(self.group_deltas):
            raise ValueError("decision shape must end with the calibrated group count")
        if not torch.all((values == 0) | (values == 1)):
            raise ValueError("group decisions must be binary")
        result = self.baseline_delta + values @ self.group_deltas
        for correction, (left, right) in zip(self.pair_corrections, self.pairs, strict=True):
            result = result + correction * values[..., left] * values[..., right]
        return result


def fit_conditional_quality(reference_nll: torch.Tensor, anchor_nll: torch.Tensor,
                            single_nll: torch.Tensor, *,
                            pairs: tuple[tuple[int, int], ...] = (),
                            pair_nll: torch.Tensor | None = None) -> ConditionalQualityFit:
    """Fit signed increments; every column must refer to the same token window.

    The anchor must run the actual fixed prefill policy, including its KV state.
    Single/pair policies change only the named decode groups. This is a finite
    interaction approximation, so unseen combinations still need validation.
    """
    reference, anchor, singles = [torch.as_tensor(x, dtype=torch.float64)
                                  for x in [reference_nll, anchor_nll, single_nll]]
    if reference.ndim != 1 or anchor.shape != reference.shape or not reference.numel():
        raise ValueError("reference and anchor must align by token window")
    if singles.ndim != 2 or singles.shape[1] != reference.numel():
        raise ValueError("single-group NLL must align by token window")
    if not all(torch.isfinite(x).all() for x in [reference, anchor, singles]):
        raise ValueError("calibration NLL must be finite")
    deltas = (singles - anchor[None]).mean(1)
    corrections = torch.empty(0, dtype=torch.float64)
    if pairs:
        if pair_nll is None:
            raise ValueError("pair NLL is required for pair corrections")
        observed = torch.as_tensor(pair_nll, dtype=torch.float64)
        if observed.shape != (len(pairs), reference.numel()) or not torch.isfinite(observed).all():
            raise ValueError("pair NLL must align by pair and token window")
        if any(a == b or min(a, b) < 0 or max(a, b) >= len(deltas) for a, b in pairs):
            raise ValueError("pair indices must name two distinct calibrated groups")
        if len({tuple(sorted(p)) for p in pairs}) != len(pairs):
            raise ValueError("pair indices must be unique")
        corrections = (observed - anchor[None]).mean(1) - torch.stack([deltas[a] + deltas[b] for a, b in pairs])
    elif pair_nll is not None:
        raise ValueError("pair NLL requires pair indices")
    return ConditionalQualityFit(float((anchor - reference).mean()), deltas, pairs, corrections)
