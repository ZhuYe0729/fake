from __future__ import annotations

import sys
from pathlib import Path


_WRAPPER_ROOT = Path(__file__).resolve().parents[1]
_STANDALONE_ROOT = _WRAPPER_ROOT / "standalone_sparse_w4a4_int4"
if str(_STANDALONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_STANDALONE_ROOT))

from sparse_w4a4_int4 import (  # noqa: E402
    SparseW4A4Int4Linear,
    SparseW4A4Int4Weight,
    can_use_sparse_w4a4_int4,
    is_2to4_sparse,
    magnitude_prune_2to4,
    pack_sparse_w4a4_int4_activation,
    quantize_sparse_w4a4_int4_weight,
    simulate_sparse_w4a4_int4_linear,
    sparse_w4a4_int4_gemm,
)

magnitude_prune_2to4_sparse_w4a4_int4 = magnitude_prune_2to4
is_2to4_sparse_w4a4_int4 = is_2to4_sparse

__all__ = [
    "SparseW4A4Int4Linear",
    "SparseW4A4Int4Weight",
    "can_use_sparse_w4a4_int4",
    "is_2to4_sparse",
    "is_2to4_sparse_w4a4_int4",
    "magnitude_prune_2to4",
    "magnitude_prune_2to4_sparse_w4a4_int4",
    "pack_sparse_w4a4_int4_activation",
    "quantize_sparse_w4a4_int4_weight",
    "simulate_sparse_w4a4_int4_linear",
    "sparse_w4a4_int4_gemm",
]
