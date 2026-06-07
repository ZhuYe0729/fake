#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CUTLASS_WRAPPER_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
for path in (REPO_ROOT, CUTLASS_WRAPPER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fake.kernels.offline_hybrid_policy import (  # noqa: E402
    LinearShapeSpec,
    ScenarioSpec,
    save_policy_json,
    select_offline_hybrid_policy,
    write_policy_csv,
)
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a generic predictor-driven offline hybrid policy.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-shapes-json", type=Path, help="JSON file containing a list of {name,n,k,count}.")
    source.add_argument("--qwen3-5-variant", default=None, help="Enumerate Qwen3.5 compressible Linear shapes.")
    parser.add_argument("--model-path", default=None, help="Model path for Qwen3.5 shape enumeration.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--kernels", nargs="+", default=None)
    parser.add_argument("--no-conversion-cost", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    linears = _load_linears(args)
    scenario = ScenarioSpec(
        batch_size=args.batch_size,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
    )
    predictor = KernelLatencyPredictor(model_root=args.model_root, kernels=args.kernels)
    policy = select_offline_hybrid_policy(
        linears,
        scenario,
        predictor,
        kernels=args.kernels,
        include_conversion_cost=not args.no_conversion_cost,
    )
    save_policy_json(policy, args.output_json)
    if args.output_csv is not None:
        write_policy_csv(policy, args.output_csv)
    selected = sum(1 for module in policy.modules if module.selected_prefill_backend is not None)
    print(
        "offline hybrid policy generated: "
        f"modules={len(policy.modules)} selected={selected} "
        f"scenario=bs{args.batch_size}_in{args.input_tokens}_out{args.output_tokens} "
        f"output={args.output_json}"
    )


def _load_linears(args: argparse.Namespace) -> list[LinearShapeSpec]:
    if args.input_shapes_json is not None:
        return _linears_from_json(args.input_shapes_json)
    return _qwen3_5_linears(args.qwen3_5_variant, args.model_path)


def _linears_from_json(path: Path) -> list[LinearShapeSpec]:
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]]
    if isinstance(payload, dict):
        rows = list(payload.get("modules", payload.get("linears", [])))
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Unsupported shapes JSON payload in {path}")
    return [
        LinearShapeSpec(
            name=str(row["name"]),
            n=int(row["n"]),
            k=int(row["k"]),
            count=int(row.get("count", 1)),
        )
        for row in rows
    ]


def _qwen3_5_linears(variant: str, model_path: str | None) -> list[LinearShapeSpec]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from fake.compression.modules import select_compressible_modules
    from fake.models.qwen3_5 import qwen3_5_model_path

    resolved_model_path = model_path or str(qwen3_5_model_path(variant))
    config = AutoConfig.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    selected = select_compressible_modules(model, "qwen3_5")
    return [
        LinearShapeSpec(
            name=info.name,
            n=int(info.module.out_features),
            k=int(info.module.in_features),
            count=1,
        )
        for info in selected
        if info.kind == "linear"
    ]


if __name__ == "__main__":
    main()
