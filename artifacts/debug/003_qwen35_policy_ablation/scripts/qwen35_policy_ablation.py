#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_main_hybrid_policy_retest import SCENARIOS, apply_policy, benchmark_model, load_model


DEFAULT_RESULT_ROOT = REPO_ROOT / "artifacts/results/main/001_hybrid_policy_retest"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/debug/003_qwen35_policy_ablation/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate Qwen3.5-9B policy differences in full-model E2E.")
    parser.add_argument("--scenario", default="normal_01", choices=SCENARIOS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = args.output_dir / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)

    manual = load_policy(args.result_root / "manual" / args.scenario / "qwen35-9b_policy.json")
    pred = load_policy(args.result_root / "pred" / args.scenario / "qwen35-9b_policy.json")
    sparse = load_policy(args.result_root / "single" / "sparse_bf16" / args.scenario / "qwen35-9b_policy.json")

    variants = {
        "single_sparse_bf16": sparse,
        "manual": manual,
        "pred": pred,
        "manual_down_to_pred": replace_groups(manual, pred, ["mlp.down_proj"]),
        "manual_kv_to_pred": replace_groups(manual, pred, ["self_attn.k_proj", "self_attn.v_proj"]),
        "manual_down_kv_to_pred": replace_groups(manual, pred, ["mlp.down_proj", "self_attn.k_proj", "self_attn.v_proj"]),
        "pred_down_to_manual": replace_groups(pred, manual, ["mlp.down_proj"]),
        "pred_kv_to_manual": replace_groups(pred, manual, ["self_attn.k_proj", "self_attn.v_proj"]),
        "pred_down_kv_to_manual": replace_groups(pred, manual, ["mlp.down_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    }

    rows: list[dict[str, Any]] = []
    for variant_name, policy in variants.items():
        policy_path = policy_dir / f"{variant_name}.json"
        policy_path.write_text(json.dumps(policy, indent=2) + "\n")
        for repeat in range(args.repeats):
            row = run_variant(args, variant_name, policy_path, repeat)
            rows.append(row)
            print(json.dumps(row, indent=2))
    write_csv(args.output_dir / "e2e_repeats.csv", rows)
    summary = summarize(rows)
    write_csv(args.output_dir / "e2e_summary.csv", summary)
    (args.output_dir / "README.md").write_text(render_readme(args, summary))


def run_variant(args: argparse.Namespace, variant_name: str, policy_path: Path, repeat: int) -> dict[str, Any]:
    dtype = torch.bfloat16
    model = load_model("qwen35-9b", dtype=dtype, gpu=args.gpu)
    report = apply_policy("qwen35-9b", model, policy_path, dtype)
    result = benchmark_model(model, SCENARIOS[args.scenario], args.gpu, args.warmup_iters)
    out_tokens = SCENARIOS[args.scenario]["output_tokens"]
    row = {
        "variant": variant_name,
        "repeat": repeat,
        "prefill_ms": result["prefill_ms"],
        "decode_x_n_ms": out_tokens * result["decode_avg_ms"],
        "decode_first_ms": result["decode_first_ms"],
        "decode_steady_ms": result["decode_steady_ms"],
        "e2e_ms": result["prefill_ms"] + out_tokens * result["decode_avg_ms"],
        "replaced_linear_count": getattr(report, "replaced_linear_count", ""),
        "backend_counts": dict(getattr(report, "backend_counts", {})),
        "policy_path": str(policy_path),
    }
    del model
    torch.cuda.empty_cache()
    return row


def load_policy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def replace_groups(base: dict[str, Any], source: dict[str, Any], groups: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    source_by_name = {module["name"]: module for module in source["modules"]}
    groups_set = set(groups)
    for module in out["modules"]:
        if module["name"] not in groups_set:
            continue
        source_module = source_by_name[module["name"]]
        module["selected_prefill_backend"] = source_module["selected_prefill_backend"]
        module["selected_decode_backend"] = source_module["selected_decode_backend"]
        module["reason"] = f"ablation_from_source:{','.join(groups)}"
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    out = []
    for variant, variant_rows in sorted(by_variant.items()):
        e2e = [float(row["e2e_ms"]) for row in variant_rows]
        prefill = [float(row["prefill_ms"]) for row in variant_rows]
        decode = [float(row["decode_x_n_ms"]) for row in variant_rows]
        out.append(
            {
                "variant": variant,
                "repeats": len(variant_rows),
                "e2e_mean_ms": statistics.mean(e2e),
                "e2e_min_ms": min(e2e),
                "e2e_max_ms": max(e2e),
                "e2e_stdev_ms": statistics.stdev(e2e) if len(e2e) > 1 else 0.0,
                "prefill_mean_ms": statistics.mean(prefill),
                "decode_x_n_mean_ms": statistics.mean(decode),
                "backend_counts": variant_rows[0]["backend_counts"],
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


def render_readme(args: argparse.Namespace, summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen3.5-9B Policy Ablation",
        "",
        f"- Scenario: `{args.scenario}` -> `{SCENARIOS[args.scenario]}`",
        f"- Repeats per variant: `{args.repeats}`",
        "",
        "| Variant | E2E mean ms | E2E min/max ms | Prefill mean ms | Decode x n mean ms | Backend counts |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summary, key=lambda r: float(r["e2e_mean_ms"])):
        lines.append(
            f"| `{row['variant']}` | {float(row['e2e_mean_ms']):.4f} | "
            f"{float(row['e2e_min_ms']):.4f}/{float(row['e2e_max_ms']):.4f} | "
            f"{float(row['prefill_mean_ms']):.4f} | {float(row['decode_x_n_mean_ms']):.4f} | "
            f"`{row['backend_counts']}` |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
