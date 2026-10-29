import torch

from daso.modeling.latency import compose_phase_latency, monotonic_calibration
from daso.modeling.quality import QualityCalibrator, build_policy_features
from daso.policy import METHODS, uniform_policy


def errors() -> dict[tuple[str, int, str, str], float]:
    output = {}
    for phase in ("prefill", "decode"):
        for bucket in range(4):
            for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                for index, method in enumerate(METHODS):
                    output[phase, bucket, kind, method] = index * 0.01
    return output


def test_phase_composition() -> None:
    assert compose_phase_latency(10, 2, 64) == 136


def test_monotonic_calibration() -> None:
    fitted = monotonic_calibration([1, 2, 3], [3, 1, 4])
    assert fitted == [2, 2, 4]


def test_quality_features_and_fit() -> None:
    names = ["model.layers.0.self_attn.qkv_proj", "model.layers.0.mlp.down_proj"]
    dense = uniform_policy("dense", names, "dense_bf16")
    quant = uniform_policy("quant", names, "dense_nvfp4")
    x = torch.stack([build_policy_features(policy, errors()) for policy in (dense, quant)])
    assert float(x[0].sum()) == 0
    fit = QualityCalibrator(steps=80).fit(
        x, torch.tensor([0.0, 0.2], dtype=torch.float64), torch.tensor([True, True]))
    prediction = fit.predict(x)
    assert prediction[1] > prediction[0]
    assert torch.all(fit.coefficients >= 0)

