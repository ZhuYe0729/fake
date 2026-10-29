import itertools

import numpy as np
import torch

from hawq_allocate import solve
from hessian_trace import trace_probe


def test_hawq_ilp_matches_exhaustive_search():
    sensitivity = np.array([[.1, .8], [.7, .2], [.4, .1]])
    latency = np.array([[4., 1.], [3., 1.], [2., 1.]])
    limit = 6.
    choice = solve(sensitivity, latency, limit)
    feasible = [candidate for candidate in itertools.product(range(2), repeat=3)
                if sum(latency[i, candidate[i]] for i in range(3)) <= limit]
    expected = min(sum(sensitivity[i, candidate[i]] for i in range(3)) for candidate in feasible)
    assert np.isclose(sensitivity[np.arange(3), choice].sum(), expected)


def test_hessian_trace_uses_loss_curvature_and_excludes_cross_blocks():
    x = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([.5], dtype=torch.float64, requires_grad=True)
    values = []
    for signs in itertools.product([-1., 1.], repeat=3):
        loss = 2 * x[0] ** 2 + 3 * x[1] ** 2 + 7 * x[0] * x[1] + 5 * y[0] ** 2 + 11 * x[0] * y[0]
        vectors = [torch.tensor(signs[:2], dtype=x.dtype), torch.tensor(signs[2:], dtype=y.dtype)]
        values.append([float(value) for value in trace_probe(loss, [x, y], vectors)])
    assert np.allclose(np.mean(values, axis=0), [10., 10.])
