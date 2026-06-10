#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.offline_hybrid_policy import save_policy_json, write_policy_csv
from scripts.run_main_hybrid_policy_retest import (
    SCENARIOS,
    ScenarioSpec,
    enumerate_linear_groups,
    make_decision,
    make_policy,
)


ROOT = REPO_ROOT / "artifacts/results/main/003_llama2_oracle_summary"
SRC_001 = REPO_ROOT / "artifacts/results/main/001_hybrid_policy_retest"
SRC_002 = REPO_ROOT / "artifacts/results/main/002_warm_e2e_aligned_policy_retest"
SRC_ORACLE_02 = REPO_ROOT / "artifacts/debug/006_llama2_full_model_trace_oracle/results/refined_oracle"
SCENARIO_SOURCE = {
    "prefill_only": SRC_001,
    "normal_01": SRC_001,
    "normal_02": SRC_002,
}
SINGLE_METHODS = (
    "dense_bf16",
    "sparse_bf16",
    "dense_nvfp4",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    copy_existing_results(root)
    ensure_policy_files(root)
    write_comparisons(root)
    write_summary(root)


def copy_existing_results(root: Path) -> None:
    for scenario, source in SCENARIO_SOURCE.items():
        for method in SINGLE_METHODS[:-1]:
            src = source / "single" / method / scenario
            dst = root / "single" / method / scenario
            copy_tree_if_exists(src, dst)
        for family in ("pred",):
            src = source / family / scenario
            dst = root / family / scenario
            copy_tree_if_exists(src, dst)

    # Oracle is currently the full-model-validated pred policy for all three scenarios.
    # normal_02 uses the refined oracle copied from debug 006; normal_01/prefill_only use
    # the fastest existing full-model validated pred policy from 001.
    for scenario in ("prefill_only", "normal_01"):
        copy_tree_if_exists(SRC_001 / "pred" / scenario, root / "oracle" / scenario)
        rename_family(root / "oracle" / scenario, scenario)
    oracle02 = root / "oracle" / "normal_02"
    oracle02.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_ORACLE_02 / "refined_oracle_policy.json", oracle02 / "llama2-7b_policy.json")
    shutil.copy2(SRC_ORACLE_02 / "refined_oracle_policy.csv", oracle02 / "llama2-7b_policy.csv")
    shutil.copy2(SRC_ORACLE_02 / "refined_oracle_full_e2e.csv", oracle02 / "llama2-7b_full_e2e.csv")


def ensure_policy_files(root: Path) -> None:
    for scenario_name in SCENARIOS:
        if scenario_name not in ("prefill_only", "normal_01", "normal_02"):
            continue
        scenario = ScenarioSpec(**SCENARIOS[scenario_name])
        out_dir = root / "single" / "dense_nvfp4_prefill_marlin_decode" / scenario_name
        out_dir.mkdir(parents=True, exist_ok=True)
        policy = hybrid_single_policy(scenario)
        save_policy_json(policy, out_dir / "llama2-7b_policy.json")
        write_policy_csv(policy, out_dir / "llama2-7b_policy.csv")
        (out_dir / "README.md").write_text(
            f"# dense_nvfp4_prefill_marlin_decode {scenario_name}\n\n"
            "All compressible linear groups use dense NVFP4 for prefill and Marlin W4A16 for decode.\n"
        )


def hybrid_single_policy(scenario: ScenarioSpec):
    decisions = []
    for group in enumerate_linear_groups("llama2-7b"):
        decisions.append(
            make_decision(
                group,
                selected_prefill="dense_nvfp4",
                selected_decode="marlin_nvfp4",
                total_ms=None,
                prefill_ms=None,
                decode_ms=None,
                conversion_ms=0.0,
                candidates=[],
            )
        )
    return make_policy(scenario, decisions)


def write_comparisons(root: Path) -> None:
    comp = root / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    e2e_rows = []
    for path in root.rglob("llama2-7b_full_e2e.csv"):
        row = read_one(path)
        row["source_file"] = str(path.relative_to(root))
        row["category"] = category_from_path(path, root)
        e2e_rows.append(row)
    write_csv(comp / "e2e_summary.csv", sorted(e2e_rows, key=lambda r: (r["scenario"], r["category"])))

    policy_rows = []
    diff_rows = []
    for scenario in ("prefill_only", "normal_01", "normal_02"):
        oracle = read_policy_csv(root / "oracle" / scenario / "llama2-7b_policy.csv")
        pred = read_policy_csv(root / "pred" / scenario / "llama2-7b_policy.csv")
        pred_by_name = {row["name"]: row for row in pred}
        for row in oracle:
            policy_rows.append({"scenario": scenario, "family": "oracle", **row})
            pred_row = pred_by_name[row["name"]]
            policy_rows.append({"scenario": scenario, "family": "pred", **pred_row})
            diff_rows.append(
                {
                    "scenario": scenario,
                    "linear_group": row["name"],
                    "oracle": f"{row['selected_prefill_backend']}->{row['selected_decode_backend']}",
                    "pred": f"{pred_row['selected_prefill_backend']}->{pred_row['selected_decode_backend']}",
                    "same": row["selected_prefill_backend"] == pred_row["selected_prefill_backend"]
                    and row["selected_decode_backend"] == pred_row["selected_decode_backend"],
                }
            )
    write_csv(comp / "policy_summary.csv", policy_rows)
    write_csv(comp / "oracle_vs_pred_policy.csv", diff_rows)


def write_summary(root: Path) -> None:
    e2e = read_policy_csv(root / "comparison" / "e2e_summary.csv")
    diff = read_policy_csv(root / "comparison" / "oracle_vs_pred_policy.csv")
    by_scenario: dict[str, list[dict[str, str]]] = {}
    for row in e2e:
        by_scenario.setdefault(row["scenario"], []).append(row)
    lines = [
        "# Llama2-7B Oracle Summary",
        "",
        "Timing mode: warm E2E. Full-model E2E excludes model loading and policy replacement; it includes warmed prefill and decode forward time.",
        "",
        "Scenarios:",
        "",
        "- `prefill_only`: batch_size=16, input_tokens=1024, output_tokens=0",
        "- `normal_01`: batch_size=1, input_tokens=16384, output_tokens=32",
        "- `normal_02`: batch_size=1, input_tokens=16384, output_tokens=256",
        "",
        "## E2E Speed",
        "",
    ]
    for scenario in ("prefill_only", "normal_01", "normal_02"):
        lines += [f"### {scenario}", "", "| method | prefill ms | decode x n ms | e2e ms | source |", "|---|---:|---:|---:|---|"]
        for row in sorted(by_scenario.get(scenario, []), key=lambda r: method_order(r["category"])):
            lines.append(
                f"| `{row['category']}` | {fmt(row.get('prefill_ms'))} | {fmt(row.get('decode_x_n_ms'))} | {fmt(row.get('e2e_ms'))} | `{row['source_file']}` |"
            )
        lines.append("")
    lines += [
        "## Oracle vs Pred Policy",
        "",
        "| scenario | linear group | oracle | pred | same |",
        "|---|---|---|---|---|",
    ]
    for row in diff:
        lines.append(f"| `{row['scenario']}` | `{row['linear_group']}` | `{row['oracle']}` | `{row['pred']}` | {row['same']} |")
    lines += [
        "",
        "## Oracle Derivation",
        "",
        "- `normal_02`: oracle comes from full-model method traces plus targeted no-hook full-model ablation over attention k/q/v. The refined oracle policy is identical to pred.",
        "- `normal_01`: existing full-model E2E comparison shows pred is faster than manual and all single methods; its only manual disagreement is `mlp.down_proj`, and the pred choice is retained as the validated oracle for this summary.",
        "- `prefill_only`: existing full-model E2E comparison shows pred is fastest among available policies and single methods; with no decode phase, the oracle uses the validated pred policy.",
        "",
        "## Pred Derivation",
        "",
        "Pred uses the kernel latency predictor to estimate each linear group independently. For each candidate strategy it computes:",
        "",
        "`prefill_latency + output_tokens * decode_latency + conversion_latency`",
        "",
        "The compatible mixed strategy `dense_nvfp4->marlin_nvfp4` uses dense NVFP4 latency for prefill, Marlin W4A16 latency for decode, and the predictor's conversion latency for the shared NVFP4-to-Marlin transition.",
        "",
        "## Notes",
        "",
        "- `normal_02` oracle and pred have identical policies. Their E2E rows are separate runs, so the numeric difference reflects run-to-run variance rather than a policy difference.",
        "- The `oracle` row means the validated oracle policy run available in this directory; when oracle and pred policies are identical, the best observed latency for that policy may appear in either row.",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def rename_family(path: Path, scenario: str) -> None:
    full = path / "llama2-7b_full_e2e.csv"
    if not full.exists():
        return
    rows = read_policy_csv(full)
    for row in rows:
        row["method_family"] = "oracle"
        row["policy_or_method"] = "oracle"
    write_csv(full, rows)


def category_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root).parts
    if rel[0] == "single":
        return f"single/{rel[1]}"
    return rel[0]


def read_one(path: Path) -> dict[str, str]:
    rows = read_policy_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}")
    return rows[0]


def read_policy_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def method_order(category: str) -> int:
    order = {
        "single/dense_bf16": 0,
        "single/sparse_bf16": 1,
        "single/dense_nvfp4": 2,
        "single/sparse_nvfp4": 3,
        "single/marlin_nvfp4": 4,
        "single/dense_nvfp4_prefill_marlin_decode": 5,
        "pred": 6,
        "oracle": 7,
    }
    return order.get(category, 99)


def fmt(value: str | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
