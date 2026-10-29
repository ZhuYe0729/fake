#!/usr/bin/env python3
"""Query the external calibrated Roofline predictor for all fused Llama actions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from daso.modeling.latency import KernelLatencyBackend


KERNELS = ("dense_bf16", "dense_nvfp4", "sparse_bf16",
           "sparse_nvfp4", "marlin_nvfp4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cutlass-wrapper-path", type=Path, required=True)
    parser.add_argument("--predictor-root", type=Path)
    parser.add_argument("--scenario", choices=("prefill_only", "prefill_decode"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", 0) or hidden // heads)
    shapes = (
        ("self_attn", "qkv_proj", hidden + 2 * kv_heads * head_dim, hidden),
        ("self_attn", "o_proj", hidden, hidden),
        ("mlp", "gate_up_proj", 2 * intermediate, hidden),
        ("mlp", "down_proj", hidden, intermediate),
    )
    phases = [("prefill", args.batch * args.input_tokens)]
    if args.scenario == "prefill_decode":
        phases.append(("decode", args.batch))
    backend = KernelLatencyBackend(args.cutlass_wrapper_path, args.predictor_root)
    rows = []
    for layer in range(int(config.num_hidden_layers)):
        for group, kind, n, k in shapes:
            name = f"model.layers.{layer}.{group}.{kind}"
            for phase, m in phases:
                predicted = backend.predict(m, n, k)
                for kernel in KERNELS:
                    action = predicted[kernel]
                    rows.append({
                        "phase": phase, "module_name": name, "layer": layer,
                        "module_type": kind, "m": m, "n": n, "k": k,
                        "kernel": kernel, "supported": action.supported,
                        "latency_ms": action.latency_ms,
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "model": str(args.model_path), "scenario": args.scenario,
        "batch": args.batch, "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "decode_steps": max(0, args.output_tokens - 1),
        "predictor_root": None if args.predictor_root is None else str(args.predictor_root),
        "module_count": int(config.num_hidden_layers) * len(shapes),
        "rows": len(rows),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
