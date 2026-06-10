#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.offline_hybrid_policy import save_policy_json, write_policy_csv
from scripts.run_main_hybrid_policy_retest import (
    KERNELS,
    SCENARIOS,
    LinearGroup,
    ScenarioSpec,
    enumerate_linear_groups,
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
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = enumerate_linear_groups("llama2-7b")
    candidates = load_candidates(args.trace_root)
    decisions = []
    ranking_rows = []
    for group in groups:
        rows = sorted(candidates[group.name], key=lambda row: float(row["projected_total_ms"]))
        best = rows[0]
        prefill_backend, decode_backend = candidate_backends(best["method"])
        decisions.append(
            make_decision(
                group,
                selected_prefill=prefill_backend,
                selected_decode=decode_backend,
                total_ms=float(best["projected_total_ms"]),
                prefill_ms=float(best["prefill_sum_ms"]) / int(group.count),
                decode_ms=float(best["decode_steady_sum_ms"]) / int(group.count),
                conversion_ms=0.0,
                candidates=[],
            )
        )
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            ranking_rows.append(out)

    scenario = ScenarioSpec(**SCENARIOS["normal_02"])
    policy = make_policy(scenario, decisions)
    save_policy_json(policy, args.out_dir / "oracle_policy.json")
    write_policy_csv(policy, args.out_dir / "oracle_policy.csv")
    write_csv(args.out_dir / "oracle_candidate_ranking.csv", ranking_rows)


def load_candidates(trace_root: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for method in METHODS:
        path = trace_root / method / "group_projection.csv"
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                out.setdefault(row["group"], []).append(row)
    return out


def candidate_backends(method: str) -> tuple[str, str]:
    if method == "dense_nvfp4_prefill_marlin_decode":
        return "dense_nvfp4", "marlin_nvfp4"
    return method, method


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
