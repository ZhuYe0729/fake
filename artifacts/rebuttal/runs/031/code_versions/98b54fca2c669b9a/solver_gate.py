"""Negative quality increments and independent choices versus exact enumeration."""
import itertools
import json
from pathlib import Path

import numpy as np

from exact_solver import solve

q = np.array([[-.03, 0.], [.02, 0.]])
t = np.array([[1., 4.], [1., 4.]])
cap = -.02
brute = [(sum(t[i, j] for i, j in enumerate(indices)), indices)
         for indices in itertools.product(range(2), repeat=2)
         if sum(q[i, j] for i, j in enumerate(indices)) <= cap + 1e-10]
result = solve(q, t, cap)
assert result['optimal'] and result['latency_ms'] == min(brute)[0]
assert result['indices'] == [0, 1] and result['quality'] < 0
path = Path(__file__).resolve().parents[1] / 'results/solver_gate.json'
path.write_text(json.dumps({'verified': True, 'negative_quality_preserved': True,
    'independent_module_choices': True, 'exact_enumeration': min(brute)[0], 'solver': result}, indent=2) + '\n')
print('Signed and independent-module solver gate passed.')
