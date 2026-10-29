"""Optional prefill preference in an additive phase quality constraint."""
import math

import torch


def combine_phase_quality(prefill: torch.Tensor, decode: torch.Tensor, *,
                          prefill_weight: float = 1.) -> torch.Tensor:
    """Return every prefill/decode joint action; no format is excluded.

    A non-unit weight is a preference score, not a recalibrated delta-NLL.
    """
    pre, dec = [torch.as_tensor(value, dtype=torch.float64) for value in [prefill, decode]]
    if pre.ndim < 1 or pre.shape[:-1] != dec.shape[:-1]:
        raise ValueError('phase costs must have matching module dimensions')
    if not math.isfinite(prefill_weight) or prefill_weight <= 0:
        raise ValueError('prefill weight must be positive and finite')
    return pre[..., :, None] * prefill_weight + dec[..., None, :]
