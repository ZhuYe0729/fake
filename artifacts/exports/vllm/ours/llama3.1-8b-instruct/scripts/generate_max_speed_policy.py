#!/usr/bin/env python3
"""Generate a predictor-only max-speed phase-heterogeneous Llama3.1 policy."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
CUTLASS_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
MODEL_PATH = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
SCENARIOS = {
    "prefill_only": {"batch": 8, "input_tokens": 2048, "output_tokens": 0, "benchmark_output_tokens": 1},
    "prefill_decode": {"batch": 16, "input_tokens": 2048, "output_tokens": 80, "benchmark_output_tokens": 80},
}
KERNELS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")
RUNTIME = {**{name: name for name in KERNELS if name != "marlin_nvfp4"}, "marlin_nvfp4": "w4a16_ours"}
CONVERSION = {"dense_nvfp4": "canonical_to_cutlass", "marlin_nvfp4": "canonical_to_marlin"}
LEGAL_PAIRS = {(x, x) for x in KERNELS} | {("dense_nvfp4", "marlin_nvfp4"), ("marlin_nvfp4", "dense_nvfp4")}

@dataclass(frozen=True)
class LinearSpec:
    name: str
    n: int
    k: int
    kind: str

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--predictor-root", type=Path)
    return p.parse_args()

def linears(config: Any) -> list[LinearSpec]:
    hidden, intermediate = int(config.hidden_size), int(config.intermediate_size)
    head_dim = int(getattr(config, "head_dim", 0) or hidden // int(config.num_attention_heads))
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    qkv_out = hidden + 2 * kv_heads * head_dim
    specs = []
    for i in range(int(config.num_hidden_layers)):
        prefix = f"model.layers.{i}"
        specs += [
            LinearSpec(f"{prefix}.self_attn.qkv_proj", qkv_out, hidden, "qkv_proj"),
            LinearSpec(f"{prefix}.self_attn.o_proj", hidden, hidden, "o_proj"),
            LinearSpec(f"{prefix}.mlp.gate_up_proj", 2 * intermediate, hidden, "gate_up_proj"),
            LinearSpec(f"{prefix}.mlp.down_proj", hidden, intermediate, "down_proj"),
        ]
    return specs

def latencies(result: Any) -> dict[str, float]:
    return {x.kernel: float(x.latency_ms) for x in result.candidates if x.kernel in KERNELS and x.supported and x.latency_ms is not None}

def select(predictor: Any, spec: LinearSpec, prefill_m: int, decode_m: int, output_tokens: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefill = latencies(predictor.predict(prefill_m, spec.n, spec.k))
    decode = latencies(predictor.predict(decode_m, spec.n, spec.k)) if output_tokens else {}
    conversions = {x.conversion: float(x.latency_ms) for x in predictor.predict_conversion(spec.n, spec.k) if x.supported and x.latency_ms is not None}
    candidates = []
    for pm in KERNELS:
        for dm in KERNELS:
            if (pm, dm) not in LEGAL_PAIRS or pm not in prefill or (output_tokens and dm not in decode):
                continue
            needed = {CONVERSION[x] for x in (pm, dm) if x in CONVERSION}
            if not needed <= conversions.keys():
                continue
            item = {"row_type": "candidate", "module_name": spec.name, "module_type": spec.kind, "n": spec.n, "k": spec.k,
                    "prefill_predictor": pm, "decode_predictor": dm, "prefill_runtime": RUNTIME[pm], "decode_runtime": RUNTIME[dm],
                    "prefill_ms": prefill[pm], "decode_ms": decode.get(dm, 0.0), "conversion_ms": sum(conversions[x] for x in needed)}
            item["total_ms"] = item["prefill_ms"] + output_tokens * item["decode_ms"] + item["conversion_ms"]
            candidates.append(item)
    if not candidates:
        raise RuntimeError(f"no legal strategy for {spec.name} ({spec.n}x{spec.k})")
    best = min(candidates, key=lambda x: x["total_ms"])
    selected = {"row_type": "selected", "module_name": spec.name, "module_type": spec.kind, "n": spec.n, "k": spec.k,
                **{f"selected_{k}": best[k] for k in ("prefill_predictor", "decode_predictor", "prefill_runtime", "decode_runtime", "prefill_ms", "decode_ms", "conversion_ms", "total_ms")}}
    return selected, candidates

def main() -> None:
    args = parse_args()
    sys.path[:0] = [str(REPO_ROOT), str(CUTLASS_ROOT), str(CUTLASS_ROOT / "modeling")]
    from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    scenario = SCENARIOS[args.scenario]
    predictor = KernelLatencyPredictor(model_root=args.predictor_root or DEFAULT_MODEL_ROOT, kernels=KERNELS)
    selected, rows, method_map = [], [], {}
    for spec in linears(config):
        choice, candidates = select(predictor, spec, scenario["batch"] * scenario["input_tokens"], scenario["batch"], scenario["output_tokens"])
        selected.append(choice); rows += candidates + [choice]
        method_map[spec.name] = {"prefill_method": choice["selected_prefill_runtime"], "decode_method": choice["selected_decode_runtime"]}
    if len(selected) != int(config.num_hidden_layers) * 4:
        raise RuntimeError(f"unexpected module count: {len(selected)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = {"default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16", "modules_to_not_convert": ["lm_head"], "method_map": method_map}
    metadata = {"model_path": str(args.model_path), "scenario": args.scenario, **scenario, "m_prefill": scenario["batch"] * scenario["input_tokens"], "m_decode": scenario["batch"], "qkv_shape": [linears(config)[0].n, linears(config)[0].k], "predictor_kernels": list(KERNELS), "runtime_method_mapping": RUNTIME, "selection_objective": "sum(prefill + output_tokens * decode + one_time_conversion)", "predicted_linear_latency_ms": sum(x["selected_total_ms"] for x in selected), "module_count": len(selected)}
    (args.output_dir / "phase_hetero_policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "policy_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "policy_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for x in rows for k in x}), extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
