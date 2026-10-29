from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from w4a4_int4 import (
    W4A4Int4Linear,
    can_use_w4a4_int4,
    quantize_w4a4_int4_weight,
    simulate_w4a4_int4_linear,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cosine": float(F.cosine_similarity(actual_f32.flatten(), expected_f32.flatten(), dim=0).item()),
    }


@pytest.mark.parametrize("m,n,k", [(1, 32, 32), (8, 64, 128), (128, 256, 256)])
def test_w4a4_int4_linear_matches_reference(m: int, n: int, k: int) -> None:
    torch.manual_seed(180)
    linear = torch.nn.Linear(k, n, bias=True, device="cuda", dtype=torch.float32).eval()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        module = W4A4Int4Linear.from_linear(linear).eval()
        actual = module(x)
        simulated = simulate_w4a4_int4_linear(x, linear.weight, linear.bias)
        bf16_expected = F.linear(x, linear.weight.to(torch.bfloat16), linear.bias.to(torch.bfloat16))

    assert actual.shape == (m, n)
    assert actual.dtype == torch.bfloat16
    sim_metrics = _error_metrics(actual, simulated)
    assert sim_metrics["cosine"] > 0.999
    assert sim_metrics["mean_abs"] < 0.05
    bf16_metrics = _error_metrics(actual, bf16_expected)
    assert bf16_metrics["cosine"] > 0.95


def test_w4a4_int4_linear_supports_batched_input() -> None:
    torch.manual_seed(181)
    linear = torch.nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        module = W4A4Int4Linear.from_linear(linear).eval()
        actual = module(x)
        simulated = simulate_w4a4_int4_linear(x, linear.weight)

    assert actual.shape == (2, 3, 64)
    metrics = _error_metrics(actual, simulated)
    assert metrics["cosine"] > 0.999


def test_w4a4_int4_quantized_weight_shapes_and_state_dict() -> None:
    torch.manual_seed(182)
    linear = torch.nn.Linear(128, 64, bias=True, device="cuda", dtype=torch.bfloat16).eval()

    with torch.no_grad():
        weight = quantize_w4a4_int4_weight(linear.weight, linear.bias)
        module = W4A4Int4Linear(weight)

    assert weight.packed_weight.shape == (64, 64)
    assert weight.packed_weight.dtype == torch.uint8
    assert weight.weight_scale.shape == (64,)
    assert weight.weight_scale.dtype == torch.float32

    state = module.state_dict()
    assert "weight" not in state
    assert "packed_weight" in state
    assert "weight_scale" in state
    assert "bias" in state


def test_w4a4_int4_reports_unsupported_shapes() -> None:
    assert not can_use_w4a4_int4(1, 33, 64)
    assert not can_use_w4a4_int4(1, 64, 33)
    assert can_use_w4a4_int4(1, 64, 64)

    bad_k = torch.nn.Linear(33, 64, bias=False, device="cuda", dtype=torch.bfloat16).eval()
    with pytest.raises(ValueError, match="divisible by 32"):
        W4A4Int4Linear.from_linear(bad_k)

    bad_n = torch.nn.Linear(64, 33, bias=False, device="cuda", dtype=torch.bfloat16).eval()
    with pytest.raises(ValueError, match="divisible by 32"):
        W4A4Int4Linear.from_linear(bad_n)


def test_w4a4_int4_linear_is_inference_only() -> None:
    linear = torch.nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        module = W4A4Int4Linear.from_linear(linear)

    module.train()
    x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="inference-only"):
        module(x)

    module.eval()
    x_requires_grad = x.detach().requires_grad_(True)
    with pytest.raises(RuntimeError, match="autograd"):
        module(x_requires_grad)
