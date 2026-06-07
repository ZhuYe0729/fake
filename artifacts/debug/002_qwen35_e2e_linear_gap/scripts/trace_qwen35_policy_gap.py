#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.compression.modules import select_compressible_modules
from fake.models.qwen3_5_kernels import QwenHybridDenseNVFP4Linear, QwenManualHybridLinear
from scripts.run_main_hybrid_policy_retest import (
    SCENARIOS,
    apply_policy,
    benchmark_model,
    load_model,
)


DEFAULT_RESULT_ROOT = REPO_ROOT / "artifacts/results/main/001_hybrid_policy_retest"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/debug/002_qwen35_e2e_linear_gap/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Qwen3.5-9B linear timing gap inside full-model E2E.")
    parser.add_argument("--scenario", default="normal_01", choices=SCENARIOS)
    parser.add_argument("--methods", nargs="+", default=["sparse_bf16", "manual", "pred"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_calls: list[dict[str, Any]] = []
    all_groups: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for method in args.methods:
        payload = run_method(args, method)
        all_calls.extend(payload["calls"])
        all_groups.extend(payload["groups"])
        method_rows.append(payload["method"])
        del payload
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "linear_calls.csv", all_calls)
    write_csv(args.output_dir / "linear_group_summary.csv", all_groups)
    write_csv(args.output_dir / "method_summary.csv", method_rows)
    (args.output_dir / "README.md").write_text(render_readme(args, method_rows, all_groups))


def run_method(args: argparse.Namespace, method: str) -> dict[str, Any]:
    dtype = torch.bfloat16
    scenario = SCENARIOS[args.scenario]
    model = load_model("qwen35-9b", dtype=dtype, gpu=args.gpu)
    policy_path = policy_for(args, method)
    report = None
    if policy_path is not None:
        report = apply_policy("qwen35-9b", model, policy_path, dtype)

    no_hook = benchmark_model(model, scenario, args.gpu, args.warmup_iters)

    target_groups = policy_groups(policy_path) if policy_path is not None else []
    recorder = LinearRecorder(method=method, batch_size=scenario["batch_size"], target_groups=target_groups)
    recorder.install(model)
    traced = traced_model_run(model, scenario, args.gpu, args.warmup_iters, recorder)
    recorder.remove()

    groups = summarize_calls(method, recorder.rows)
    method_row = {
        "method": method,
        "policy_path": "" if policy_path is None else str(policy_path),
        "no_hook_prefill_ms": no_hook["prefill_ms"],
        "no_hook_decode_x_n_ms": scenario["output_tokens"] * no_hook["decode_avg_ms"],
        "no_hook_e2e_ms": no_hook["prefill_ms"] + scenario["output_tokens"] * no_hook["decode_avg_ms"],
        "traced_prefill_ms": traced["prefill_ms"],
        "traced_decode_x_n_ms": sum(traced["decode_times"]),
        "traced_e2e_ms": traced["prefill_ms"] + sum(traced["decode_times"]),
        "traced_linear_sum_ms": sum(float(row["total_ms"]) for row in groups),
        "replaced_linear_count": "" if report is None else getattr(report, "replaced_linear_count", ""),
        "backend_counts": "" if report is None else dict(getattr(report, "backend_counts", {})),
    }
    print(json.dumps(method_row, indent=2))
    del model
    return {"calls": recorder.rows, "groups": groups, "method": method_row}


def policy_for(args: argparse.Namespace, method: str) -> Path | None:
    if method == "dense_bf16":
        return None
    if method in {"manual", "pred"}:
        return args.result_root / method / args.scenario / "qwen35-9b_policy.json"
    return args.result_root / "single" / method / args.scenario / "qwen35-9b_policy.json"


@torch.inference_mode()
def traced_model_run(
    model: nn.Module,
    scenario: dict[str, int],
    gpu: int,
    warmup_iters: int,
    recorder: "LinearRecorder",
) -> dict[str, Any]:
    device = f"cuda:{gpu}"
    for _ in range(warmup_iters):
        recorder.enabled = False
        ids = torch.randint(0, 1000, (scenario["batch_size"], min(32, scenario["input_tokens"])), device=device)
        _ = model(ids, use_cache=False)
    torch.cuda.synchronize()
    recorder.clear()
    recorder.enabled = True

    input_ids = torch.randint(0, 1000, (scenario["batch_size"], scenario["input_tokens"]), device=device)
    recorder.region = "prefill"
    prefill_ms, out = time_forward(lambda: model(input_ids, use_cache=scenario["output_tokens"] > 0))
    decode_times = []
    if scenario["output_tokens"] > 0:
        past_key_values = out.past_key_values
        next_token = torch.randint(0, 1000, (scenario["batch_size"], 1), device=device)
        for step in range(scenario["output_tokens"]):
            recorder.region = "decode"
            recorder.decode_step = step
            ms, out = time_forward(lambda: model(next_token, past_key_values=past_key_values, use_cache=True))
            decode_times.append(ms)
            past_key_values = out.past_key_values
            next_token = torch.randint(0, 1000, (scenario["batch_size"], 1), device=device)
    recorder.enabled = False
    return {"prefill_ms": prefill_ms, "decode_times": decode_times}


def time_forward(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


class LinearRecorder:
    def __init__(self, *, method: str, batch_size: int, target_groups: list[str]) -> None:
        self.method = method
        self.batch_size = int(batch_size)
        self.target_groups = set(target_groups)
        self.region = ""
        self.decode_step = -1
        self.enabled = False
        self.rows: list[dict[str, Any]] = []
        self._starts: dict[int, tuple[torch.cuda.Event, int, str]] = {}
        self._handles = []

    def install(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if ".backends." in name:
                continue
            group = normalize_group_name(name)
            if self.target_groups and group not in self.target_groups:
                continue
            if not is_replaced_linear_module(module):
                continue
            self._handles.append(module.register_forward_pre_hook(self._pre(name)))
            self._handles.append(module.register_forward_hook(self._post(name)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.rows.clear()
        self._starts.clear()

    def _pre(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not self.enabled:
                return
            x = inputs[0]
            m = int(x.reshape(-1, x.shape[-1]).size(0)) if isinstance(x, torch.Tensor) else -1
            input_shape = tuple(x.shape) if isinstance(x, torch.Tensor) else ()
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._starts[id(module)] = (start, m, backend_for(module, m, self.batch_size), input_shape)

        return hook

    def _post(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self.enabled:
                return
            start, m, backend, input_shape = self._starts.pop(id(module))
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            self.rows.append(
                {
                    "method": self.method,
                    "name": name,
                    "group": normalize_group_name(name),
                    "module_type": type(module).__name__,
                    "region": self.region,
                    "decode_step": self.decode_step if self.region == "decode" else "",
                    "m": m,
                    "input_shape": "x".join(map(str, input_shape)),
                    "backend": backend,
                    "elapsed_ms": float(start.elapsed_time(end)),
                }
            )

        return hook


def backend_for(module: nn.Module, m: int, batch_size: int) -> str:
    if isinstance(module, QwenManualHybridLinear):
        return module.decode_backend if m <= module.decode_m_threshold else module.prefill_backend
    if isinstance(module, QwenHybridDenseNVFP4Linear):
        return module.decode_backend if m <= module.marlin_m_threshold else module.prefill_backend
    name = type(module).__name__
    if "SparseBF16" in name:
        return "sparse_bf16"
    if "SparseNVFP4" in name:
        return "sparse_nvfp4"
    if "Marlin" in name:
        return "marlin_nvfp4"
    if "NVFP4" in name:
        return "dense_nvfp4"
    return "dense_bf16"


def is_replaced_linear_module(module: nn.Module) -> bool:
    if isinstance(module, (QwenManualHybridLinear, QwenHybridDenseNVFP4Linear)):
        return True
    name = type(module).__name__
    return any(token in name for token in ("PaddedSparseBF16", "PaddedSparseNVFP4", "MarlinNVFP4", "NVFP4Linear"))


def policy_groups(policy_path: Path) -> list[str]:
    payload = json.loads(policy_path.read_text())
    return [str(row["name"]) for row in payload.get("modules", [])]


def normalize_group_name(name: str) -> str:
    name = re.sub(r"^(model\.)?language_model\.layers\.\d+\.", "", name)
    name = re.sub(r"^(model\.)?layers\.\d+\.", "", name)
    return name


def summarize_calls(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["group"]), str(row["region"]), str(row["backend"]))
        grouped.setdefault(key, []).append(float(row["elapsed_ms"]))
    out = []
    for (group, region, backend), values in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "group": group,
                "region": region,
                "backend": backend,
                "calls": len(values),
                "total_ms": sum(values),
                "avg_ms": sum(values) / len(values),
                "first_ms": values[0],
                "max_ms": max(values),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_readme(args: argparse.Namespace, method_rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen3.5-9B E2E Linear Gap Trace",
        "",
        f"- Scenario: `{args.scenario}` -> `{SCENARIOS[args.scenario]}`",
        "- `no_hook_*`: normal full-model E2E timing, same model object before hooks.",
        "- `traced_*`: full-model E2E with CUDA event hooks on every compressible linear; this is only for attribution and is expected to be slower.",
        "- `traced_linear_sum_ms`: sum of all measured compressible linear module forwards during the traced run.",
        "",
        "## Method Summary",
        "",
        "| Method | No-hook E2E | Traced E2E | Traced linear sum | Backend counts |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method']}` | {float(row['no_hook_e2e_ms']):.4f} | "
            f"{float(row['traced_e2e_ms']):.4f} | {float(row['traced_linear_sum_ms']):.4f} | "
            f"`{row['backend_counts']}` |"
        )
    lines.extend(["", "## Largest Group Totals", "", "| Method | Group | Region | Backend | Calls | Total ms | First ms | Max ms |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    largest = sorted(groups, key=lambda r: float(r["total_ms"]), reverse=True)[:40]
    for row in largest:
        lines.append(
            f"| `{row['method']}` | `{row['group']}` | `{row['region']}` | `{row['backend']}` | "
            f"{row['calls']} | {float(row['total_ms']):.4f} | {float(row['first_ms']):.4f} | {float(row['max_ms']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
