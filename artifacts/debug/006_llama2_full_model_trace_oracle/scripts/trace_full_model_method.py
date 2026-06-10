#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.compression.modules import select_compressible_modules
from fake.kernels.offline_hybrid_policy import save_policy_json, write_policy_csv
from scripts.run_main_hybrid_policy_retest import (
    KERNELS,
    MODELS,
    SCENARIOS,
    LinearGroup,
    ScenarioSpec,
    apply_policy,
    enumerate_linear_groups,
    load_model,
    make_decision,
    make_policy,
)


METHODS = (
    "dense_bf16",
    "sparse_bf16",
    "dense_nvfp4",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    scenario_dict = SCENARIOS["normal_02"]
    scenario = ScenarioSpec(**scenario_dict)
    device = f"cuda:{args.gpu}"

    model = load_model("llama2-7b", dtype=dtype, gpu=args.gpu)
    selected = select_compressible_modules(model, "llama")
    target_names = [item.name for item in selected if item.kind == "linear"]
    del selected

    policy_path = None
    report = None
    if args.method != "dense_bf16":
        policy = method_policy(args.method, scenario)
        policy_path = args.out_dir / f"{args.method}_policy.json"
        save_policy_json(policy, policy_path)
        write_policy_csv(policy, args.out_dir / f"{args.method}_policy.csv")
        report = apply_policy("llama2-7b", model, policy_path, dtype)

    rows = trace_model(model, target_names, scenario_dict, device, args.decode_steps)
    write_csv(args.out_dir / "raw_linear_trace.csv", rows)
    projections = summarize_projection(rows, args.method, scenario_dict["output_tokens"], args.decode_steps)
    write_csv(args.out_dir / "group_projection.csv", projections)
    write_csv(args.out_dir / "module_projection.csv", summarize_module_projection(rows, args.method, scenario_dict["output_tokens"], args.decode_steps))
    (args.out_dir / "report.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "scenario": scenario_dict,
                "decode_steps_traced": args.decode_steps,
                "policy_path": "" if policy_path is None else str(policy_path),
                "replacement_report": None
                if report is None
                else {
                    "replaced_linear_count": report.replaced_linear_count,
                    "skipped_linear_count": report.skipped_linear_count,
                    "backend_counts": report.backend_counts,
                },
                "target_linear_count": len(target_names),
                "raw_trace_rows": len(rows),
            },
            indent=2,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def method_policy(method: str, scenario: ScenarioSpec):
    groups = enumerate_linear_groups("llama2-7b")
    decisions = []
    for group in groups:
        if method == "dense_nvfp4_prefill_marlin_decode":
            prefill_backend = "dense_nvfp4"
            decode_backend = "marlin_nvfp4"
        else:
            prefill_backend = method
            decode_backend = method
        decisions.append(
            make_decision(
                group,
                selected_prefill=prefill_backend,
                selected_decode=decode_backend,
                total_ms=None,
                prefill_ms=None,
                decode_ms=None,
                conversion_ms=0.0,
                candidates=[],
            )
        )
    return make_policy(scenario, decisions)


def trace_model(model: nn.Module, target_names: list[str], scenario: dict[str, int], device: str, decode_steps: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    handles = []
    state = {"call_index": 0, "decode_step": -1}

    def pre_hook(name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            x = inputs[0]
            token_count = int(x.reshape(-1, x.shape[-1]).shape[0])
            phase = "prefill" if token_count == scenario["input_tokens"] else ("decode" if token_count == 1 else f"warmup_m{token_count}")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            record = {
                "call_index": state["call_index"],
                "decode_step": state["decode_step"] if phase == "decode" else "",
                "module": name,
                "group": normalize_group_name(name),
                "phase": phase,
                "token_count": token_count,
                "module_type": type(module).__name__,
                "start": start,
                "end": end,
            }
            state["call_index"] += 1
            start.record()
            setattr(module, "_trace_record", record)
        return hook

    def post_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        record = getattr(module, "_trace_record", None)
        if record is None:
            return
        record["end"].record()
        events.append(record)
        setattr(module, "_trace_record", None)

    for name in target_names:
        module = module_by_name(model, name)
        handles.append(module.register_forward_pre_hook(pre_hook(name)))
        handles.append(module.register_forward_hook(post_hook))

    ids = torch.randint(0, 1000, (scenario["batch_size"], min(32, scenario["input_tokens"])), device=device)
    _ = model(ids, use_cache=False)
    torch.cuda.synchronize()
    events.clear()
    state["call_index"] = 0

    input_ids = torch.randint(0, 1000, (scenario["batch_size"], scenario["input_tokens"]), device=device)
    out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    past_key_values = out.past_key_values
    for step in range(decode_steps):
        state["decode_step"] = step
        next_token = torch.randint(0, 1000, (scenario["batch_size"], 1), device=device)
        out = model(next_token, past_key_values=past_key_values, use_cache=True)
        torch.cuda.synchronize()
        past_key_values = out.past_key_values

    for handle in handles:
        handle.remove()

    rows = []
    for item in events:
        rows.append(
            {
                "call_index": item["call_index"],
                "decode_step": item["decode_step"],
                "module": item["module"],
                "group": item["group"],
                "phase": item["phase"],
                "token_count": item["token_count"],
                "module_type": item["module_type"],
                "latency_ms": item["start"].elapsed_time(item["end"]),
            }
        )
    return rows


def summarize_projection(rows: list[dict[str, Any]], method: str, output_tokens: int, decode_steps: int) -> list[dict[str, Any]]:
    by_group_phase: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group_phase[(str(row["group"]), str(row["phase"]))].append(row)
    out = []
    for group in sorted({str(row["group"]) for row in rows}):
        prefill_rows = by_group_phase.get((group, "prefill"), [])
        decode_rows = by_group_phase.get((group, "decode"), [])
        step_sums = decode_step_sums(decode_rows)
        first = step_sums[0] if step_sums else 0.0
        steady_values = step_sums[1:] if len(step_sums) > 1 else step_sums
        steady = sum(steady_values) / max(len(steady_values), 1)
        prefill = sum(float(row["latency_ms"]) for row in prefill_rows)
        out.append(
            {
                "method": method,
                "group": group,
                "prefill_sum_ms": prefill,
                "decode_first_sum_ms": first,
                "decode_steady_sum_ms": steady,
                "decode_steps_traced": decode_steps,
                "projected_total_ms": prefill + first + max(output_tokens - 1, 0) * steady,
                "prefill_backend": candidate_backends(method)[0],
                "decode_backend": candidate_backends(method)[1],
            }
        )
    return out


def summarize_module_projection(rows: list[dict[str, Any]], method: str, output_tokens: int, decode_steps: int) -> list[dict[str, Any]]:
    modules = sorted({str(row["module"]) for row in rows})
    out = []
    for module in modules:
        sub = [row for row in rows if row["module"] == module]
        prefill = sum(float(row["latency_ms"]) for row in sub if row["phase"] == "prefill")
        decode_rows = [row for row in sub if row["phase"] == "decode"]
        step_sums = decode_step_sums(decode_rows)
        first = step_sums[0] if step_sums else 0.0
        steady_values = step_sums[1:] if len(step_sums) > 1 else step_sums
        steady = sum(steady_values) / max(len(steady_values), 1)
        group = normalize_group_name(module)
        out.append(
            {
                "method": method,
                "module": module,
                "group": group,
                "prefill_ms": prefill,
                "decode_first_ms": first,
                "decode_steady_ms": steady,
                "decode_steps_traced": decode_steps,
                "projected_total_ms": prefill + first + max(output_tokens - 1, 0) * steady,
                "prefill_backend": candidate_backends(method)[0],
                "decode_backend": candidate_backends(method)[1],
            }
        )
    return out


def decode_step_sums(rows: list[dict[str, Any]]) -> list[float]:
    by_step: dict[int, float] = defaultdict(float)
    for row in rows:
        by_step[int(row["decode_step"])] += float(row["latency_ms"])
    return [by_step[step] for step in sorted(by_step)]


def candidate_backends(method: str) -> tuple[str, str]:
    if method == "dense_nvfp4_prefill_marlin_decode":
        return "dense_nvfp4", "marlin_nvfp4"
    return method, method


def normalize_group_name(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[-2:])


def module_by_name(model: nn.Module, name: str) -> nn.Module:
    current: nn.Module = model
    for part in name.split("."):
        current = getattr(current, part)
    return current


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
