from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marlin_int4 import (
    MarlinInt4Linear,
    can_use_marlin_int4,
    quantize_marlin_int4_weight,
    simulate_marlin_int4_linear,
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


@pytest.mark.parametrize("m,n,k", [(1, 256, 128), (8, 512, 1024), (65, 256, 512)])
def test_marlin_int4_linear_matches_reference(m: int, n: int, k: int) -> None:
    torch.manual_seed(190)
    linear = torch.nn.Linear(k, n, bias=True, device="cuda", dtype=torch.float16).eval()
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        module = MarlinInt4Linear.from_linear(linear).eval()
        actual = module(x)
        simulated = simulate_marlin_int4_linear(x, linear.weight, linear.bias)
        fp16_expected = F.linear(x, linear.weight, linear.bias)

    assert actual.shape == (m, n)
    assert actual.dtype == torch.float16
    sim_metrics = _error_metrics(actual, simulated)
    assert sim_metrics["cosine"] > 0.999
    assert sim_metrics["mean_abs"] < 0.05
    assert _error_metrics(actual, fp16_expected)["cosine"] > 0.95


def test_marlin_int4_linear_supports_batched_input() -> None:
    torch.manual_seed(191)
    linear = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.float16).eval()
    x = torch.randn(2, 3, 256, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        module = MarlinInt4Linear.from_linear(linear).eval()
        actual = module(x)
        simulated = simulate_marlin_int4_linear(x, linear.weight)

    assert actual.shape == (2, 3, 256)
    assert _error_metrics(actual, simulated)["cosine"] > 0.999


def test_marlin_int4_supports_per_column_groupsize() -> None:
    torch.manual_seed(192)
    linear = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.float16).eval()
    x = torch.randn(4, 256, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        module = MarlinInt4Linear.from_linear(linear, groupsize=-1).eval()
        actual = module(x)
        simulated = simulate_marlin_int4_linear(x, linear.weight, groupsize=-1)

    assert module.groupsize == -1
    assert _error_metrics(actual, simulated)["cosine"] > 0.999


def test_marlin_int4_quantized_weight_shapes_and_state_dict() -> None:
    torch.manual_seed(193)
    linear = torch.nn.Linear(256, 512, bias=True, device="cuda", dtype=torch.float16).eval()

    with torch.no_grad():
        weight = quantize_marlin_int4_weight(linear.weight, linear.bias)
        module = MarlinInt4Linear(weight)

    assert weight.packed_weight.shape == (16, 1024)
    assert weight.packed_weight.dtype == torch.int32
    assert weight.weight_scale.shape == (2, 512)
    assert weight.weight_scale.dtype == torch.float16

    state = module.state_dict()
    assert "weight" not in state
    assert "packed_weight" in state
    assert "weight_scale" in state
    assert "zero_points" in state
    assert "bias" in state
    assert "workspace" not in state

    reloaded = MarlinInt4Linear(weight)
    reloaded.load_state_dict(state)


def test_marlin_int4_reports_unsupported_shapes_and_groups() -> None:
    assert not can_use_marlin_int4(1, 255, 128)
    assert not can_use_marlin_int4(1, 256, 127)
    assert can_use_marlin_int4(1, 256, 128)

    bad_k = torch.nn.Linear(127, 256, bias=False, device="cuda", dtype=torch.float16).eval()
    with pytest.raises(ValueError, match="divisible by 128"):
        MarlinInt4Linear.from_linear(bad_k)

    bad_n = torch.nn.Linear(128, 255, bias=False, device="cuda", dtype=torch.float16).eval()
    with pytest.raises(ValueError, match="divisible by 256"):
        MarlinInt4Linear.from_linear(bad_n)

    with pytest.raises(ValueError, match="groupsize"):
        MarlinInt4Linear.from_linear(
            torch.nn.Linear(128, 256, bias=False, device="cuda", dtype=torch.float16).eval(),
            groupsize=64,
        )


def test_marlin_int4_linear_is_inference_only() -> None:
    linear = torch.nn.Linear(128, 256, bias=False, device="cuda", dtype=torch.float16).eval()
    with torch.no_grad():
        module = MarlinInt4Linear.from_linear(linear)

    module.train()
    x = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="inference-only"):
        module(x)

    module.eval()
    x_requires_grad = x.detach().requires_grad_(True)
    with pytest.raises(RuntimeError, match="autograd"):
        module(x_requires_grad)
