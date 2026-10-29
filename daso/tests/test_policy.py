import pytest

from daso.policy import policy_sha256, uniform_policy, validate_policy


def test_uniform_policy_is_stable() -> None:
    names = ["model.layers.0.self_attn.qkv_proj", "model.layers.0.mlp.down_proj"]
    policy = uniform_policy("p00", names, "dense_bf16")
    validate_policy(policy, layers=1)
    assert policy_sha256(policy) == policy_sha256(policy)


def test_rejects_unknown_method() -> None:
    policy = uniform_policy("p00", ["model.layers.0.mlp.down_proj"], "dense_bf16")
    policy["method_map"]["model.layers.0.mlp.down_proj"]["decode_method"] = "direct_prune"
    with pytest.raises(ValueError):
        validate_policy(policy)

