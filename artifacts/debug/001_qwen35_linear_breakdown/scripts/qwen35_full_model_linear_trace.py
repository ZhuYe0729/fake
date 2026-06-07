#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from scripts.bench_qwen3_5_swh_e2e import convert_inplace, load_dense


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/debug/001_qwen35_linear_breakdown/full_model_trace"
DEFAULT_POLICY_JSON = (
    REPO_ROOT / "artifacts/results/benchmarks/hybrid/pred/normal_01/qwen3_5_9b_normal_01_policy.json"
)
DEFAULT_LAYERS = [
    "language_model.layers.0.linear_attn.in_proj_qkv",
    "language_model.layers.0.linear_attn.in_proj_z",
    "language_model.layers.0.mlp.gate_proj",
    "language_model.layers.0.mlp.down_proj",
    "language_model.layers.3.self_attn.q_proj",
    "language_model.layers.3.self_attn.o_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace selected Qwen3.5 linears inside full model E2E forward.")
    parser.add_argument("--variant", default="9B")
    parser.add_argument("--methods", nargs="+", default=["sparse_bf16", "predictor_hybrid"])
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY_JSON)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    method_payloads: list[dict[str, Any]] = []
    for method in args.methods:
        print(f"\n== Full-model trace: {method} ==")
        payload = run_method(args, method, dtype, device)
        method_payloads.append(payload)
        all_rows.extend(payload["trace_rows"])
        summary_rows.extend(payload["summary_rows"])
        del payload
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "linear_trace.csv", all_rows)
    write_csv(args.output_dir / "linear_trace_summary.csv", summary_rows)
    (args.output_dir / "linear_trace.json").write_text(json.dumps(method_payloads, indent=2) + "\n")
    (args.output_dir.parent / "FULL_MODEL_TRACE.md").write_text(render_readme(args, method_payloads))
    print(f"\nwrote {args.output_dir}")


def run_method(args: argparse.Namespace, method: str, dtype: torch.dtype, device: torch.device) -> dict[str, Any]:
    model = load_dense(args.variant, dtype)
    convert_start = time.perf_counter()
    report = convert_inplace(
        model,
        method,
        dtype,
        policy_json=str(args.policy_json) if method == "predictor_hybrid" else None,
    )
    torch.cuda.synchronize()
    convert_ms = (time.perf_counter() - convert_start) * 1000.0

    recorder = LinearTraceRecorder(
        method=method,
        layers=args.layers,
        batch_size=args.batch_size,
    )
    recorder.install(model)

    with torch.inference_mode():
        for _ in range(args.warmup_iters):
            ids = torch.randint(0, 1000, (args.batch_size, 32), device=device)
            _ = model(ids)
        torch.cuda.synchronize()
        recorder.clear()

        prefill_ids = torch.randint(0, 1000, (args.batch_size, args.input_tokens), device=device)
        recorder.current_region = "prefill"
        prefill_ms, prefill_out = time_model_forward(lambda: model(prefill_ids, use_cache=True))

        past_key_values = prefill_out.past_key_values
        next_token = torch.randint(0, 1000, (args.batch_size, 1), device=device)
        decode_times = []
        for step in range(args.output_tokens):
            recorder.current_region = "decode"
            recorder.current_decode_step = step
            decode_ms, out = time_model_forward(
                lambda: model(next_token, past_key_values=past_key_values, use_cache=True)
            )
            decode_times.append(decode_ms)
            past_key_values = out.past_key_values
            next_token = torch.randint(0, 1000, (args.batch_size, 1), device=device)

    recorder.remove()
    trace_rows = recorder.rows
    summary_rows = summarize_trace(
        trace_rows,
        method=method,
        prefill_ms=prefill_ms,
        decode_times=decode_times,
        convert_ms=convert_ms,
    )
    payload = {
        "method": method,
        "metadata": {
            "model": f"Qwen3.5-{args.variant}",
            "batch_size": args.batch_size,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "m_prefill": args.batch_size * args.input_tokens,
            "m_decode": args.batch_size,
            "warmup_iters": args.warmup_iters,
            "gpu": torch.cuda.get_device_name(device),
            "convert_ms": convert_ms,
            "report_replaced_linear_count": getattr(report, "replaced_linear_count", ""),
            "report_skipped_linear_count": getattr(report, "skipped_linear_count", ""),
            "report_backend_counts": getattr(report, "backend_counts", ""),
            "prefill_ms": prefill_ms,
            "decode_first_ms": decode_times[0],
            "decode_avg_ms": sum(decode_times) / len(decode_times),
            "decode_x_n_ms": sum(decode_times),
            "e2e_ms": prefill_ms + sum(decode_times),
        },
        "trace_rows": trace_rows,
        "summary_rows": summary_rows,
    }
    print(
        f"  method={method} convert={convert_ms:.2f}ms prefill={prefill_ms:.2f}ms "
        f"decode_x_n={sum(decode_times):.2f}ms e2e={prefill_ms + sum(decode_times):.2f}ms"
    )
    del model
    return payload


class LinearTraceRecorder:
    def __init__(self, *, method: str, layers: list[str], batch_size: int) -> None:
        self.method = method
        self.layers = set(layers)
        self.batch_size = int(batch_size)
        self.rows: list[dict[str, Any]] = []
        self.handles = []
        self.current_region = ""
        self.current_decode_step = -1
        self._starts: dict[int, tuple[torch.cuda.Event, int]] = {}

    def install(self, model: nn.Module) -> None:
        modules = dict(model.named_modules())
        missing = []
        for name in self.layers:
            module = modules.get(name)
            if module is None:
                missing.append(name)
                continue
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._post_hook(name)))
        if missing:
            raise ValueError(f"Missing target modules after conversion: {missing}")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.rows.clear()
        self._starts.clear()

    def _pre_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            x = inputs[0]
            m = int(x.reshape(-1, x.shape[-1]).size(0)) if isinstance(x, torch.Tensor) else -1
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._starts[id(module)] = (event, m)

        return hook

    def _post_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            start, m = self._starts.pop(id(module))
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            region = self.current_region
            phase = "decode" if m <= self.batch_size else "prefill"
            self.rows.append(
                {
                    "method": self.method,
                    "layer": name,
                    "module_type": type(module).__name__,
                    "region": region,
                    "phase_from_m": phase,
                    "decode_step": self.current_decode_step if region == "decode" else "",
                    "m": m,
                    "elapsed_ms": elapsed_ms,
                }
            )

        return hook


def time_model_forward(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), result


def summarize_trace(
    rows: list[dict[str, Any]],
    *,
    method: str,
    prefill_ms: float,
    decode_times: list[float],
    convert_ms: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["layer"]), str(row["region"]))
        grouped.setdefault(key, []).append(float(row["elapsed_ms"]))
    out = []
    for (layer, region), values in sorted(grouped.items()):
        total = sum(values)
        avg = total / len(values)
        out.append(
            {
                "method": method,
                "layer": layer,
                "region": region,
                "calls": len(values),
                "total_ms": total,
                "avg_ms": avg,
                "first_ms": values[0],
                "max_ms": max(values),
                "model_prefill_ms": prefill_ms,
                "model_decode_x_n_ms": sum(decode_times),
                "model_e2e_ms": prefill_ms + sum(decode_times),
                "convert_ms": convert_ms,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def render_readme(args: argparse.Namespace, payloads: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen3.5-9B Full-Model Linear Trace",
        "",
        "## Scenario",
        "",
        f"- Workload: `batch_size={args.batch_size}, input_tokens={args.input_tokens}, output_tokens={args.output_tokens}`",
        f"- Warmup before traced run: `{args.warmup_iters}`",
        "- Timing method: CUDA events in forward pre/post hooks on the real replaced modules during full model prefill/decode.",
        "",
        "## Full E2E",
        "",
        "| Method | Convert ms | Prefill ms | Decode first ms | Decode x n ms | E2E ms | Backend counts |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for payload in payloads:
        meta = payload["metadata"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{payload['method']}`",
                    f"{meta['convert_ms']:.4f}",
                    f"{meta['prefill_ms']:.4f}",
                    f"{meta['decode_first_ms']:.4f}",
                    f"{meta['decode_x_n_ms']:.4f}",
                    f"{meta['e2e_ms']:.4f}",
                    f"`{meta['report_backend_counts']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Selected Linear Summary",
            "",
            "| Method | Layer | Region | Calls | Total ms | Avg ms | First ms | Max ms |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for payload in payloads:
        for row in payload["summary_rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['method']}`",
                        f"`{row['layer']}`",
                        f"`{row['region']}`",
                        str(row["calls"]),
                        f"{row['total_ms']:.4f}",
                        f"{row['avg_ms']:.4f}",
                        f"{row['first_ms']:.4f}",
                        f"{row['max_ms']:.4f}",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `full_model_trace/linear_trace.csv`: per-call hook records.",
            "- `full_model_trace/linear_trace_summary.csv`: per-layer phase summary.",
            "- `full_model_trace/linear_trace.json`: full structured payload.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
