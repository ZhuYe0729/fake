from typing import Tuple, Optional
import math

import torch

from .quant_args import QuantizationFormat, QuantizationGranularity, QuantizationObserver, ScalePrecision
from .quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX, FP4_SCALE, get_quantization_fns, get_quantization_range, cast_to_eBm0
from ..utils.helpers import split_dim

# Utility function for inversion.
def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


class Quantizer:

    def __init__(
        self, 
        bits: int, 
        symmetric: bool = True,
        format: str = "int",
        granularity: str = "channel",
        observer: str = "minmax",
        dim: int = -1,
        group_size: Optional[int] = None,
        scale_precision: str = "fp16",
        scale_min_clip: Optional[float] = None,
        scale_factor: float = 1.0,
        scale_remap: str = "none",
        scale_remap_gamma: float = 1.0,
        scale_remap_mu: float = 5.0,
    ):
        # Sanity checks
        if format in ["fp", "nvfp", "mxfp"]:
            assert symmetric, "Only symmetric quantization is supported for floating point formats."

        if granularity == "group":
            assert group_size is not None, "Group size must be specified when granularity is 'group'."
        else:
            assert group_size is None, "Group size must be None when granularity is not 'group'."

        self.bits = bits
        self.symmetric = symmetric
        self.format = QuantizationFormat(format)
        self.granularity = QuantizationGranularity(granularity)
        self.observer = QuantizationObserver(observer)
        self.scale_precision = ScalePrecision(scale_precision)
        self.dim = dim
        self.group_size = group_size
        self.scale_min_clip = scale_min_clip
        self.scale_factor = scale_factor
        self.scale_remap = scale_remap
        self.scale_remap_gamma = scale_remap_gamma
        self.scale_remap_mu = scale_remap_mu

        if self.scale_remap == "none" and self.scale_factor != 1.0:
            self.scale_remap = "linear"

        if self.format == QuantizationFormat.NVFP:
            assert self.group_size in (16, 32), (
                f"NVFP4 requires group_size=16 or 32, got {self.group_size}."
            )

        self.quant_fn, self.dequant_fn, self.quant_dequant_fn = get_quantization_fns(
            format=self.format,
            bits=self.bits,
        )

        self.q_min, self.q_max = get_quantization_range(
            format=self.format,
            bits=self.bits,
            symmetric=self.symmetric,
        )
        
        # Global scale is 3 for MXFP quantization
        if self.format == QuantizationFormat.MXFP:
            self.global_scale = torch.tensor([3.0], dtype=torch.float32)
        else:
            self.global_scale = torch.tensor([float("inf")], dtype=torch.float32)
        # Scale tracking is needed only for E4M3 scale quantization
        self._track_global_scale = (self.scale_precision == ScalePrecision.E4M3)

    def _reshape_before_quantization(
        self, 
        x: torch.Tensor, 
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.group_size:
            dim = x.ndim - 1 if self.dim == -1 else self.dim
            num_groups = x.shape[dim] // self.group_size
            x = split_dim(x, num_groups, dim)
            if scales is not None:
                scales = scales.unsqueeze(dim + 1)
            if zeros is not None:
                zeros = zeros.unsqueeze(dim + 1)
        return x, scales, zeros

    # ------------------------------------------------------------------
    # Non-linear remapping in normalized FP4 space [-6, 6] → [-6, 6]
    # Both preserve endpoints: f(0)=0, f(±6)=±6, and are invertible.
    # ------------------------------------------------------------------

    def _power_remap(self, u: torch.Tensor) -> torch.Tensor:
        """f(u) = sign(u) * 6 * (|u|/6)^gamma — power companding."""
        gamma = self.scale_remap_gamma
        return torch.sign(u) * 6.0 * (torch.abs(u) / 6.0).pow(gamma)

    def _power_remap_inv(self, v: torch.Tensor) -> torch.Tensor:
        gamma = self.scale_remap_gamma
        return torch.sign(v) * 6.0 * (torch.abs(v) / 6.0).pow(1.0 / gamma)

    def _mulaw_remap(self, u: torch.Tensor) -> torch.Tensor:
        """f(u) = sign(u) * 6 * log(1 + mu*|u|/6) / log(1 + mu) — mu-law."""
        mu = self.scale_remap_mu
        inv_log_mu1 = 1.0 / math.log(1.0 + mu)
        return torch.sign(u) * 6.0 * torch.log1p(mu * torch.abs(u) / 6.0) * inv_log_mu1

    def _mulaw_remap_inv(self, v: torch.Tensor) -> torch.Tensor:
        mu = self.scale_remap_mu
        log_mu1 = math.log(1.0 + mu)
        return torch.sign(v) * (6.0 / mu) * (torch.exp(torch.abs(v) * log_mu1 / 6.0) - 1.0)

    def _get_remap_fns(self):
        """Return (forward, inverse) remap callables for current mode."""
        if self.scale_remap in ("power", "auto"):
            return self._power_remap, self._power_remap_inv
        if self.scale_remap == "mulaw":
            return self._mulaw_remap, self._mulaw_remap_inv
        return None, None

    def calibrate_rescale_gamma(self, weight: torch.Tensor):
        """Derive optimal power-companding gamma from pruned weight distribution.

        Uses the linear relationship derived from optimal compander theory:
            gamma* = 0.295 * (mu / sigma) + 0.977
        where mu, sigma are parameters of a symmetric Gaussian mixture fitted
        to the normalized weight distribution (|w| / scale).

        After calibration, self.scale_remap is set to "power" and
        self.scale_remap_gamma is set to the derived value.
        """
        w = weight.detach().float()

        if self.group_size is not None and self.group_size > 0:
            gs = self.group_size
            if w.shape[-1] % gs != 0:
                raise ValueError(
                    f"Weight dim {w.shape[-1]} not divisible by group_size {gs}"
                )
            w_grouped = w.reshape(-1, gs)
            scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            w_norm = (w_grouped / scales).reshape(-1)
        else:
            scale = w.abs().amax().clamp(min=1e-12)
            w_norm = (w / scale).reshape(-1)

        # Only use non-zero elements (pruned zeros are uninformative)
        w_abs = w_norm.abs()
        nonzero_mask = w_abs > 1e-8
        w_abs = w_abs[nonzero_mask]

        if w_abs.numel() < 32:
            self.scale_remap = "power"
            self.scale_remap_gamma = 1.0
            return

        # Fit symmetric Gaussian mixture: peak location (mu) and spread (sigma)
        mu_hat = w_abs.median().item()
        sigma_hat = w_abs.std().item()
        if sigma_hat < 1e-8:
            sigma_hat = 1e-8

        gamma_star = 0.295 * (mu_hat / sigma_hat) + 0.977
        gamma_star = max(0.3, min(gamma_star, 3.0))

        self.scale_remap = "power"
        self.scale_remap_gamma = gamma_star

    def calibrate_rescale_alpha(self, weight: torch.Tensor, lambd: float = 2.8):
        """Derive optimal linear rescale alpha from pruned weight distribution.

        Uses the effective-range model derived from MSE distortion analysis:
            alpha* = 6 / (mu_hat + lambda * sigma_hat)
        where mu_hat and sigma_hat are the median and std of the non-zero
        normalized weight magnitudes, and lambda is the coverage coefficient.

        Unlike power companding, linear rescale is fully hardware-compatible:
        it is equivalent to modifying the per-group scale (s' = s / alpha),
        so Tensor Core block-scaled GEMM computes the correct result directly.

        After calibration, self.scale_remap is set to "linear" and
        self.scale_factor is set to the derived alpha.
        """
        w = weight.detach().float()

        if self.group_size is not None and self.group_size > 0:
            gs = self.group_size
            if w.shape[-1] % gs != 0:
                raise ValueError(
                    f"Weight dim {w.shape[-1]} not divisible by group_size {gs}"
                )
            w_grouped = w.reshape(-1, gs)
            scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            w_norm = (w_grouped / scales).reshape(-1)
        else:
            scale = w.abs().amax().clamp(min=1e-12)
            w_norm = (w / scale).reshape(-1)

        w_abs = w_norm.abs()
        nonzero_mask = w_abs > 1e-8
        w_abs = w_abs[nonzero_mask]

        if w_abs.numel() < 32:
            self.scale_remap = "linear"
            self.scale_factor = 1.0
            return

        mu_hat = w_abs.median().item()
        sigma_hat = w_abs.std().item()
        if sigma_hat < 1e-8:
            sigma_hat = 1e-8

        e_eff = mu_hat + lambd * sigma_hat
        if e_eff < 1e-8:
            e_eff = 1e-8

        alpha_star = 6.0 / e_eff
        alpha_star = max(0.8, min(alpha_star, 5.0))

        self.scale_remap = "linear"
        self.scale_factor = alpha_star

    def get_quantization_params(
        self, 
        x: torch.Tensor,
        # MSE observer quantization params
        scale_search_iters: int = 100,
        max_scale_shrink_factor: float = 0.80,
        error_norm: float = 2.4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get scale and zero point for an input tensor.
        """
        dim = x.ndim - 1 if self.dim == -1 else self.dim
        if self.granularity == QuantizationGranularity.GROUP:
            reduce_dim = dim + 1
        elif self.granularity == QuantizationGranularity.CHANNEL:
            reduce_dim = dim
        else:
            reduce_dim = None
        x, _, _ = self._reshape_before_quantization(x)

        x_min = x.amin(dim=reduce_dim, keepdim=True)
        x_max = x.amax(dim=reduce_dim, keepdim=True)

        if self.symmetric:
            scales = 2 * torch.maximum(-x_min, x_max) / (self.q_max - self.q_min)
            zeros =  torch.zeros_like(x_min)
        else:
            scales = (x_max - x_min) / (self.q_max - self.q_min)
            zeros = -(x_min / scales).round()

        if self.observer == QuantizationObserver.MSE:
            init_scales = scales.clone() 
            best_quantization_error = torch.full(x.shape[:-1], float("inf"), device=x.device, dtype=x.dtype)

            for i in range(scale_search_iters):
                scale_shrink_factor = 1 - i * max_scale_shrink_factor / scale_search_iters
                candidate_scales = scale_shrink_factor * init_scales
                candidate_zeros = torch.zeros_like(x_min) if self.symmetric else -(x_min / candidate_scales).round() 
                q = self.quant_fn(x, candidate_scales, candidate_zeros, self.q_min, self.q_max)
                x_reconstructed = self.dequant_fn(q, candidate_scales, candidate_zeros)
                quantization_error = (x - x_reconstructed).abs_().pow_(error_norm).sum(dim=-1)

                if (quantization_error < best_quantization_error).any():
                    improved_ids = torch.where(quantization_error < best_quantization_error)
                    best_quantization_error[improved_ids] = quantization_error[improved_ids]
                    scales[improved_ids] = candidate_scales[improved_ids]
                    if not self.symmetric:
                        zeros[improved_ids] = candidate_zeros[improved_ids]

        # Reshape back
        if self.group_size:
            x = x.flatten(dim, dim + 1)
            scales = scales.squeeze(dim + 1)
            if zeros is not None:
                zeros = zeros.squeeze(dim + 1)

        if self.scale_precision == ScalePrecision.E4M3:
            with torch.no_grad():
                if self._track_global_scale:
                    current_global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(x.abs().max().to(torch.float32).view(1))
                    if not current_global_scale:
                        raise ValueError(f"Current global scale is not finite: {current_global_scale}\n")
                    # Update global scale using min of current and computed scale
                    self.global_scale = torch.minimum(self.global_scale.to(x.device), current_global_scale)
                    
                    if not self.global_scale.isfinite():
                        raise ValueError(f"Global scale is not finite: {self.global_scale}\n")
                    
                # Clamp, convert to fp8, convert back, and rescale in one chain
                scales = (scales * self.global_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX) \
                    .to(torch.float8_e4m3fn) \
                    .to(torch.float32) \
                    .mul(get_reciprocal(self.global_scale)) \
                    .to(x.dtype)
        
        elif self.scale_precision == ScalePrecision.E8M0:
            # Inspired by quantize_tseng (see https://github.com/IST-DASLab/Quartet/blob/main/notebooks/benchmark_mxfp4.ipynb)
            # NOTE (in quartet x.abs().max() is defined as a scale insteaf of x.abs().max() / q_max )
            scales = cast_to_eBm0(FP4_E2M1_MAX * scales, ebits=8, emax=2) / FP4_SCALE

        # Set scales to 1 if zero
        scales[scales == 0] = 1

        if scales.isnan().any():
            raise ValueError(f"Scales are not finite.")
      
        return scales, zeros
        
    def quantize(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        remap_fn, _ = self._get_remap_fns()
        if remap_fn is not None:
            x_r, s_r, z_r = self._reshape_before_quantization(x, scales, zeros)
            v = remap_fn(x_r / s_r)
            return self.quant_fn(v * s_r, s_r, z_r, self.q_min, self.q_max).reshape(original_shape)
        if self.scale_remap == "linear":
            x = x * self.scale_factor
        q = self.quant_fn(
            *self._reshape_before_quantization(x, scales, zeros), 
            self.q_min, 
            self.q_max
        ).reshape(original_shape)
        return q

    def dequantize(self, q: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = q.shape
        _, inv_fn = self._get_remap_fns()
        if inv_fn is not None:
            q_r, s_r, _ = self._reshape_before_quantization(q, scales, zeros)
            return (inv_fn(q_r) * s_r).reshape(original_shape)
        result = self.dequant_fn(
            *self._reshape_before_quantization(q, scales, zeros), 
        ).reshape(original_shape)
        if self.scale_remap == "linear":
            result = result / self.scale_factor
        return result
    
    def __call__(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        remap_fn, inv_fn = self._get_remap_fns()
        if remap_fn is not None:
            x_r, s_r, z_r = self._reshape_before_quantization(x, scales, zeros)
            v = remap_fn(x_r / s_r)
            q = self.quant_fn(v * s_r, s_r, z_r, self.q_min, self.q_max)
            xq = (inv_fn(q) * s_r).reshape(original_shape)
            return x + (xq - x).detach()
        if self.scale_remap == "linear":
            x_scaled = x * self.scale_factor
            xq = self.quant_dequant_fn(
                *self._reshape_before_quantization(x_scaled, scales, zeros), 
                self.q_min, 
                self.q_max
            ).reshape(original_shape)
            xq = xq / self.scale_factor
            return x + (xq - x).detach()
        q = self.quant_dequant_fn(
            *self._reshape_before_quantization(x, scales, zeros), 
            self.q_min, 
            self.q_max
        ).reshape(original_shape)
        return q
