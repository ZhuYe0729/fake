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


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.compression.modules import select_compressible_modules
from fake.models.llama_kernels import replace_linear_with_llama_predictor_hybrid
from scripts.run_main_hybrid_policy_retest import MODELS, SCENARIOS, load_model


GROUP_SUFFIXES = (
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steady-decode-steps", type=int, default=8)
    return parser.parse_args()


def group_name(module_name: str) -> str:
    for suffix in GROUP_SUFFIXES:
        if module_name.endswith(suffix):
            return suffix
    raise ValueError(module_name)


def module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    current: torch.nn.Module = model
    for part in name.split("."):
        current = getattr(current, part)
    return current


def current_phase(token_count: int, scenario: dict[str, int]) -> str:
    if token_count == scenario["input_tokens"]:
        return "prefill"
    if token_count == 1:
        return "decode"
    return f"warmup_m{token_count}"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    scenario = SCENARIOS["normal_02"]

    model = load_model("llama2-7b", dtype=torch.bfloat16, gpu=args.gpu)
    selected = select_compressible_modules(model, "llama")
    target_names = [item.name for item in selected if item.kind == "linear" and item.name.endswith(GROUP_SUFFIXES)]
    del selected
    report = replace_linear_with_llama_predictor_hybrid(model, policy_path=args.policy, activation_dtype=torch.bfloat16)
    name_to_group = {name: group_name(name) for name in target_names}

    events: list[dict[str, Any]] = []
    handles = []
    state = {"call_index": 0}

    def pre_hook(name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            x = inputs[0]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            token_count = int(x.reshape(-1, x.shape[-1]).shape[0])
            record = {
                "call_index": state["call_index"],
                "module": name,
                "group": name_to_group[name],
                "phase": current_phase(token_count, scenario),
                "token_count": token_count,
                "module_type": type(module).__name__,
                "start": start,
                "end": end,
            }
            state["call_index"] += 1
            start.record()
            setattr(module, "_trace_record", record)
        return hook

    def post_hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
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

    for _ in range(1):
        ids = torch.randint(0, 1000, (scenario["batch_size"], min(32, scenario["input_tokens"])), device=device)
        _ = model(ids, use_cache=False)
    torch.cuda.synchronize()
    events.clear()
    state["call_index"] = 0

    input_ids = torch.randint(0, 1000, (scenario["batch_size"], scenario["input_tokens"]), device=device)
    out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    past_key_values = out.past_key_values
    for _ in range(1 + args.steady_decode_steps):
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
                "label": args.label,
                "call_index": item["call_index"],
                "module": item["module"],
                "group": item["group"],
                "phase": item["phase"],
                "token_count": item["token_count"],
                "module_type": item["module_type"],
                "latency_ms": item["start"].elapsed_time(item["end"]),
            }
        )
    raw_path = args.out_dir / f"{args.label}_raw_linear_trace.csv"
    write_csv(raw_path, rows)

    summary = summarize(rows, args.label)
    write_csv(args.out_dir / f"{args.label}_group_summary.csv", summary)
    (args.out_dir / f"{args.label}_report.json").write_text(
        json.dumps(
            {
                "label": args.label,
                "policy": str(args.policy),
                "scenario": scenario,
                "replacement_report": {
                    "replaced_linear_count": report.replaced_linear_count,
                    "skipped_linear_count": report.skipped_linear_count,
                    "backend_counts": report.backend_counts,
                },
                "raw_trace": str(raw_path),
            },
            indent=2,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def summarize(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    module_types: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["group"], row["phase"])
        grouped[key].append(float(row["latency_ms"]))
        module_types[key].add(str(row["module_type"]))

    out = []
    for (group, phase), values in sorted(grouped.items()):
        first = values[0]
        steady_values = values[1:] if phase == "decode" else values
        out.append(
            {
                "label": label,
                "group": group,
                "phase": phase,
                "calls": len(values),
                "sum_ms": sum(values),
                "avg_ms": sum(values) / len(values),
                "first_ms": first,
                "steady_avg_ms": sum(steady_values) / max(len(steady_values), 1),
                "module_types": ";".join(sorted(module_types[(group, phase)])),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
