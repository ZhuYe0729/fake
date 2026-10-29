#!/usr/bin/env python3
"""Summarize prefill/decode decomposition for one benchmark scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def trace_overhead(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    data = load_json(path)
    trace = data["trace"]
    sched = [e["time"] for e in trace if e["event"] == "schedule_decode_copy"]
    enter = [e["time"] for e in trace if e["event"] == "enter_decode"]
    first_decode = [e["time"] for e in trace if e["event"] == "apply_decode"]
    sync_end = [e for e in trace if e["event"] == "sync_decode_copy_end"]
    out: dict[str, float | None] = {
        "trace_num_layers": data.get("num_layers"),
        "trace_num_scheduled": data.get("num_scheduled"),
    }
    if sched and enter:
        out["async_schedule_window_s"] = max(sched) - min(sched)
        out["async_first_schedule_to_decode_s"] = enter[0] - min(sched)
    if enter and first_decode:
        out["decode_boundary_wait_s"] = first_decode[0] - enter[0]
    if sync_end:
        out["sync_copy_s"] = sync_end[-1]["duration_s"]
    return out


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    labels = [
        "dense_nvfp4",
        "w4a16_ours",
        "phase_switch_overlap",
        "phase_switch_sync",
    ]
    rows = []
    full_by_label = {}
    for label in labels:
        base = f"{label}_{args.scenario}"
        prefill = load_json(result_dir / f"{base}_o1.json")
        full = load_json(result_dir / f"{base}_o{args.output_len}.json")
        full_by_label[label] = full
        prefill_s = prefill["generate_s"]
        total_s = full["generate_s"]
        decode_s = max(total_s - prefill_s, 0.0)
        prompt_tokens = full["prompt_tokens"]
        decode_tokens = max(full["output_tokens"] - prefill["output_tokens"], 0)
        row = {
            "label": label,
            "prefill_s_approx": prefill_s,
            "decode_s_approx": decode_s,
            "total_s": total_s,
            "prefill_tokens_per_s": prompt_tokens / prefill_s,
            "decode_tokens_per_s": decode_tokens / decode_s
            if decode_s > 0 else None,
            "total_tokens_per_s": full["total_tokens_per_s"],
            "prompt_tokens": prompt_tokens,
            "decode_tokens_for_decode_phase": decode_tokens,
        }
        row.update(
            trace_overhead(result_dir /
                           f"{base}_o{args.output_len}_trace.json"))
        rows.append(row)

    overlap = next(r for r in rows if r["label"] == "phase_switch_overlap")
    sync = next(r for r in rows if r["label"] == "phase_switch_sync")
    sync_copy = sync.get("sync_copy_s")
    boundary_wait = overlap.get("decode_boundary_wait_s")
    overhead = {
        "non_overlap_switch_s": sync_copy,
        "overlap_boundary_wait_s": boundary_wait,
        "hidden_switch_s": (sync_copy - boundary_wait)
        if sync_copy is not None and boundary_wait is not None else None,
        "sync_total_minus_overlap_total_s":
        sync["total_s"] - overlap["total_s"],
    }

    summary = {
        "scenario": args.scenario,
        "method": "prefill approximated by output_len=1; decode is full run minus output_len=1 run",
        "rows": rows,
        "switch_overhead": overhead,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    lines = [
        "# Decomposed Phase Switch Benchmark",
        "",
        f"scenario: `{args.scenario}`",
        "",
        "Prefill is approximated by the same prompt batch with `output_len=1`; "
        "decode is `output_len=N` total time minus the `output_len=1` run.",
        "",
        "| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decode_tps = row["decode_tokens_per_s"]
        lines.append(
            "| {label} | {prefill_s_approx:.4f} | {decode_s_approx:.4f} | "
            "{total_s:.4f} | {prefill_tokens_per_s:.2f} | "
            "{decode_tps} | {total_tokens_per_s:.2f} |".format(
                decode_tps=f"{decode_tps:.2f}" if decode_tps else "n/a",
                **row))
    lines.extend([
        "",
        "## Switch overhead",
        "",
        f"- non-overlap switch copy: {overhead['non_overlap_switch_s']:.4f}s",
        f"- exposed boundary wait with overlap: {overhead['overlap_boundary_wait_s']:.4f}s",
        f"- hidden by overlap: {overhead['hidden_switch_s']:.4f}s",
        "- sync total minus overlap total: "
        f"{overhead['sync_total_minus_overlap_total_s']:.4f}s",
        "",
    ])
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
