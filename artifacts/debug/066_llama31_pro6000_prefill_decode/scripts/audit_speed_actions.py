#!/usr/bin/env python3
"""Freeze Llama3.1 phase shapes and roofline-predicted legal kernel actions."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
from scenario import BATCH, INPUT_TOKENS, OUTPUT_TOKENS, EXP, MODEL, CUTLASS
sys.path[:0] = [str(ROOT), str(CUTLASS), str(CUTLASS / "modeling")]
from modeling.kernel_predictor import KernelLatencyPredictor  # noqa: E402

KERNELS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")
CONVERSIONS = {"dense_nvfp4": "canonical_to_cutlass", "marlin_nvfp4": "canonical_to_marlin"}


def specs(config: AutoConfig) -> list[tuple[str, int, str, int, int]]:
    hidden, intermediate = int(config.hidden_size), int(config.intermediate_size)
    head_dim = hidden // int(config.num_attention_heads)
    qkv_width = hidden + 2 * int(config.num_key_value_heads) * head_dim
    return [(f"model.layers.{layer}.{group}.{name}", layer, name, n, k)
            for layer in range(int(config.num_hidden_layers))
            for group, name, n, k in (("self_attn", "qkv_proj", qkv_width, hidden),
                                      ("self_attn", "o_proj", hidden, hidden),
                                      ("mlp", "gate_up_proj", 2 * intermediate, hidden),
                                      ("mlp", "down_proj", hidden, intermediate))]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-root", type=Path, required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    predictor = KernelLatencyPredictor(model_root=args.predictor_root, kernels=KERNELS)
    rows = []
    for phase, m in (("prefill", BATCH * INPUT_TOKENS), ("decode", BATCH)):
        for name, layer, module_type, n, k in specs(config):
            candidates = {item.kernel: item for item in predictor.predict(m, n, k).candidates}
            conversions = {item.conversion: item for item in predictor.predict_conversion(n, k)}
            for kernel in KERNELS:
                item, conversion = candidates[kernel], conversions.get(CONVERSIONS.get(kernel, ""))
                rows.append({"phase": phase, "module_name": name, "layer": layer, "module_type": module_type,
                             "m": m, "n": n, "k": k, "kernel": kernel, "supported": bool(item.supported),
                             "latency_ms": item.latency_ms, "source": item.source, "reason": item.reason,
                             "prediction_status": item.prediction_status, "prediction_error": item.prediction_error,
                             "conversion": CONVERSIONS.get(kernel, ""),
                             "conversion_supported": bool(conversion and conversion.supported) if kernel in CONVERSIONS else True,
                             "conversion_ms": conversion.latency_ms if conversion else 0.0,
                             "conversion_reason": conversion.reason if conversion else ""})
    protocol = {"scenario": "prefill_decode", "batch": BATCH, "input_tokens": INPUT_TOKENS,
                "output_tokens": OUTPUT_TOKENS, "m_prefill": BATCH * INPUT_TOKENS,
                "m_decode": BATCH, "methods": list(KERNELS),
                "logical_runtime_mapping": {"marlin_nvfp4": "w4a16_ours"}, "model_path": str(MODEL),
                "module_count": len(specs(config)), "predictor_root": str(args.predictor_root.resolve()),
                "one_time_conversion_excluded_from_solver": True,
                "config": {key: int(getattr(config, key)) for key in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")}}
    (EXP / "architecture_manifest.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (EXP / "speed").mkdir(parents=True, exist_ok=True)
    with (EXP / "speed/action_support.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    bad = [row for row in rows if not row["supported"] or not row["conversion_supported"]]
    print(json.dumps({"actions": len(rows), "unsupported": len(bad), "module_count": len(specs(config))}, indent=2))


if __name__ == "__main__":
    main()
