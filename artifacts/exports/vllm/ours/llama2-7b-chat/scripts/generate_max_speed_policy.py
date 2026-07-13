#!/usr/bin/env python3
"""Generate a predictor-only max-speed phase-heterogeneous Llama policy."""

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
MODEL_PATH = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
SCENARIOS = {
    "prefill_only": {"batch": 8, "input_tokens": 2048, "output_tokens": 0, "benchmark_output_tokens": 1},
    "prefill_decode": {"batch": 16, "input_tokens": 2048, "output_tokens": 80, "benchmark_output_tokens": 80},
}
PREDICTOR_KERNELS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")
RUNTIME_METHOD = {**{name: name for name in PREDICTOR_KERNELS if name != "marlin_nvfp4"}, "marlin_nvfp4": "w4a16_ours"}
NVFP4_CONVERSION = {"dense_nvfp4": "canonical_to_cutlass", "marlin_nvfp4": "canonical_to_marlin"}
LEGAL_PAIRS = {(left, right) for left in PREDICTOR_KERNELS for right in PREDICTOR_KERNELS if left == right}
LEGAL_PAIRS.update({("dense_nvfp4", "marlin_nvfp4"), ("marlin_nvfp4", "dense_nvfp4")})


@dataclass(frozen=True)
class LinearSpec:
    name: str
    n: int
    k: int
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictor-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path[:0] = [str(REPO_ROOT), str(CUTLASS_ROOT), str(CUTLASS_ROOT / "modeling")]
    from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    scenario = SCENARIOS[args.scenario]
    predictor = KernelLatencyPredictor(model_root=args.predictor_root or DEFAULT_MODEL_ROOT, kernels=PREDICTOR_KERNELS)
    specs = llama_vllm_linears(config)
    m_prefill = scenario["batch"] * scenario["input_tokens"]
    m_decode = scenario["batch"]
    rows, method_map = [], {}
    for spec in specs:
        result = select_strategy(predictor, spec, m_prefill, m_decode, scenario["output_tokens"])
        rows.extend(result.pop("candidate_rows"))
        rows.append(result)
        method_map[spec.name] = {
            "prefill_method": result["selected_prefill_runtime"],
            "decode_method": result["selected_decode_runtime"],
        }
    selected = [row for row in rows if row.get("row_type") == "selected"]
    policy = {
        "default_prefill_method": "dense_bf16",
        "default_decode_method": "dense_bf16",
        "modules_to_not_convert": ["lm_head"],
        "method_map": method_map,
    }
    metadata = {
        "model_path": str(args.model_path), "scenario": args.scenario, **scenario,
        "m_prefill": m_prefill, "m_decode": m_decode, "predictor_kernels": list(PREDICTOR_KERNELS),
        "runtime_method_mapping": RUNTIME_METHOD,
        "selection_objective": "sum(prefill + output_tokens * decode + one_time_conversion)",
        "predicted_linear_latency_ms": sum(float(row["selected_total_ms"]) for row in selected),
        "module_count": len(selected),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "phase_hetero_policy.json", policy)
    write_json(args.output_dir / "policy_metadata.json", metadata)
    write_csv(args.output_dir / "policy_candidates.csv", rows)
    print(json.dumps(metadata, indent=2))


def llama_vllm_linears(config: Any) -> list[LinearSpec]:
    layers, hidden, intermediate = int(config.num_hidden_layers), int(config.hidden_size), int(config.intermediate_size)
    specs = []
    for index in range(layers):
        prefix = f"model.layers.{index}"
        specs.extend((
            LinearSpec(f"{prefix}.self_attn.qkv_proj", 3 * hidden, hidden, "qkv_proj"),
            LinearSpec(f"{prefix}.self_attn.o_proj", hidden, hidden, "o_proj"),
            LinearSpec(f"{prefix}.mlp.gate_up_proj", 2 * intermediate, hidden, "gate_up_proj"),
            LinearSpec(f"{prefix}.mlp.down_proj", hidden, intermediate, "down_proj"),
        ))
    return specs


def select_strategy(predictor: Any, spec: LinearSpec, m_prefill: int, m_decode: int, output_tokens: int) -> dict[str, Any]:
    prefill = candidate_latency(predictor.predict(m_prefill, spec.n, spec.k))
    decode = candidate_latency(predictor.predict(m_decode, spec.n, spec.k)) if output_tokens else {}
    conversions = {item.conversion: item.latency_ms for item in predictor.predict_conversion(spec.n, spec.k) if item.supported and item.latency_ms is not None}
    candidates, candidate_rows = [], []
    for prefill_method in PREDICTOR_KERNELS:
        for decode_method in PREDICTOR_KERNELS:
            if (prefill_method, decode_method) not in LEGAL_PAIRS or prefill_method not in prefill:
                continue
            if output_tokens and decode_method not in decode:
                continue
            needed = {NVFP4_CONVERSION[method] for method in (prefill_method, decode_method) if method in NVFP4_CONVERSION}
            if not needed.issubset(conversions):
                continue
            conversion_ms = sum(float(conversions[name]) for name in needed)
            decode_ms = 0.0 if not output_tokens else float(decode[decode_method])
            total = float(prefill[prefill_method]) + output_tokens * decode_ms + conversion_ms
            candidate = {"row_type": "candidate", "module_name": spec.name, "module_type": spec.kind, "n": spec.n, "k": spec.k,
                         "prefill_predictor": prefill_method, "decode_predictor": decode_method,
                         "prefill_runtime": RUNTIME_METHOD[prefill_method], "decode_runtime": RUNTIME_METHOD[decode_method],
                         "prefill_ms": float(prefill[prefill_method]), "decode_ms": decode_ms,
                         "conversion_ms": conversion_ms, "total_ms": total}
            candidates.append(candidate)
            candidate_rows.append(candidate)
    if not candidates:
        raise RuntimeError(f"no legal predicted strategy for {spec.name} ({spec.n}x{spec.k})")
    best = min(candidates, key=lambda item: item["total_ms"])
    return {"row_type": "selected", "module_name": spec.name, "module_type": spec.kind, "n": spec.n, "k": spec.k,
            "selected_prefill_predictor": best["prefill_predictor"], "selected_decode_predictor": best["decode_predictor"],
            "selected_prefill_runtime": best["prefill_runtime"], "selected_decode_runtime": best["decode_runtime"],
            "selected_prefill_ms": best["prefill_ms"], "selected_decode_ms": best["decode_ms"],
            "selected_conversion_ms": best["conversion_ms"], "selected_total_ms": best["total_ms"], "candidate_rows": candidate_rows}


def candidate_latency(selection: Any) -> dict[str, float]:
    return {item.kernel: float(item.latency_ms) for item in selection.candidates if item.kernel in PREDICTOR_KERNELS and item.supported and item.latency_ms is not None}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
