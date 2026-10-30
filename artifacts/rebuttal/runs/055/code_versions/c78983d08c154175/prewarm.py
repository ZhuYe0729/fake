"""Compile each required runtime extension serially and check its numerics."""
from common import WRAPPER
from graph_bindings import install

install()

import pytest

tests = [
    'test_nvfp4_linear.py::test_nvfp4_linear_matches_bf16_shape_and_reasonable_error',
    'test_sparse_nvfp4_linear.py::test_sparse_nvfp4_linear_prune_true_matches_simulated_reference',
    'test_sparse_bf16_linear.py::test_sparse_bf16_linear_prune_true_matches_pruned_bf16_reference',
    'test_marlin_nvfp4_linear.py::test_marlin_nvfp4_linear_matches_weight_only_reference',
]
raise SystemExit(pytest.main(['-q', '--tb=short', *[str(WRAPPER / 'tests' / test) for test in tests]]))
