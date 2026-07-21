#!/usr/bin/env python3
"""Run the established DP Pareto solver with 054 canonical-sparse features."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"))
LEGACY = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/solve_real_vllm_pareto.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_local_errors() -> dict[tuple[int, str, str], float]:
    rows = read_csv(EXPERIMENT / "local_errors/module_method_errors.csv")
    parts = {"qkv_proj": {"q_proj", "k_proj", "v_proj"}, "o_proj": {"o_proj"},
             "gate_up_proj": {"gate_proj", "up_proj"}, "down_proj": {"down_proj"}}
    methods = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
    aliases = {"w4a16_ours": "dense_nvfp4"}
    result = {}
    for layer in range(32):
        for fused, members in parts.items():
            for method in methods:
                if method == "dense_bf16":
                    result[layer, fused, method] = 0.0
                    continue
                values = [float(row["local_rel_mse"]) for row in rows
                          if int(row["layer"]) == layer and row["module_type"] in members
                          and row["method"] == aliases.get(method, method)]
                if not values:
                    raise RuntimeError(f"missing canonical feature {layer} {fused} {method}")
                result[layer, fused, method] = sum(values) / len(values)
    return result


def main() -> None:
    sys.path.insert(0, str(LEGACY.parent))
    namespace = {"__name__": "canonical_solver_import", "__file__": str(LEGACY)}
    exec(compile(LEGACY.read_text(), str(LEGACY), "exec"), namespace)
    namespace["model_root"] = lambda _model: EXPERIMENT
    namespace["local_errors"] = canonical_local_errors
    namespace["main"]()


if __name__ == "__main__":
    main()
