#!/usr/bin/env python3
"""Join local predicted/exact sums with existing 058 phase-vLLM E2E values."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/059_llama31_prefill_speed_decomposition"
SOURCE = ROOT / "artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/llama31_8b_instruct"
ACTION = ROOT / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto/action_support.csv"
METHOD_TO_KERNEL = {"dense_bf16": "dense_bf16", "dense_nvfp4": "dense_nvfp4", "sparse_bf16": "sparse_bf16", "sparse_nvfp4": "sparse_nvfp4", "w4a16_ours": "marlin_nvfp4"}
LABELS = ("p00", "p01", "p02", "p03", "p04", "point_000", "point_003", "point_005", "point_007", "point_008", "point_009", "point_010", "point_011", "point_012", "point_014", "bridge_dense_nvfp4_072", "bridge_dense_nvfp4_088", "bridge_dense_nvfp4_104", "bridge_dense_nvfp4_120")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def policy_path(label: str) -> Path:
    if len(label) == 3 and label.startswith("p") and label[1:].isdigit():
        return SOURCE / "policies/prefill_only" / f"{label}.json"
    return SOURCE / "pareto/policies" / f"{label}.json"


def main() -> None:
    exact_rows = read(DEBUG / "exact_micro/targeted_profile.csv")
    exact = {(int(row["m"]), int(row["n"]), int(row["k"]), row["kernel"]): float(row["latency_ms"])
             for row in exact_rows if row["status"] == "pass" and row["latency_ms"]}
    action_rows = read(ACTION)
    predicted = {(int(row["m"]), int(row["n"]), int(row["k"]), row["kernel"]): row
                 for row in action_rows if row["supported"] == "True"}
    closure = {row["policy_id"]: row for row in read(SOURCE / "pareto/closure_summary.csv")}
    anchors = {row["policy_id"]: row for row in read(SOURCE / "speed/calibration/calibration.csv")}
    detail: list[dict[str, object]] = []
    for key, row in sorted(predicted.items()):
        if key not in exact:
            continue
        value = exact[key]
        detail.append({"m": key[0], "n": key[1], "k": key[2], "kernel": key[3], "predictor_source": row["source"],
                       "predicted_local_ms": float(row["latency_ms"]), "exact_local_ms": value,
                       "local_error_ms": float(row["latency_ms"]) - value,
                       "local_error_pct": 100 * (float(row["latency_ms"]) - value) / value})
    report = DEBUG / "report"; report.mkdir(parents=True, exist_ok=True)
    with (report / "per_shape_local_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)

    policies: list[dict[str, object]] = []
    for label in LABELS:
        path = policy_path(label)
        if not path.exists():
            continue
        policy = json.loads(path.read_text())
        predicted_sum = exact_sum = 0.0
        counts: defaultdict[str, int] = defaultdict(int)
        for name, pair in policy["method_map"].items():
            method = pair["prefill_method"]; kernel = METHOD_TO_KERNEL[method]; counts[method] += 1
            parts = name.split(".")
            if ".self_attn.qkv_proj" in name: shape = (16384, 6144, 4096)
            elif ".self_attn.o_proj" in name: shape = (16384, 4096, 4096)
            elif ".mlp.gate_up_proj" in name: shape = (16384, 28672, 4096)
            elif ".mlp.down_proj" in name: shape = (16384, 4096, 14336)
            else: raise ValueError(name)
            key = (*shape, kernel)
            predicted_sum += float(predicted[key]["latency_ms"])
            exact_sum += exact[key]
        if label in closure:
            e2e = float(closure[label]["measured_e2e_ms"])
        elif label in anchors:
            e2e = float(anchors[label]["e2e_median_ms"])
        else:
            e2e = None
        policies.append({"policy_id": label, "predicted_local_sum_ms": predicted_sum, "exact_local_sum_ms": exact_sum,
                         "local_prediction_error_ms": predicted_sum - exact_sum,
                         "local_prediction_error_pct": 100 * (predicted_sum - exact_sum) / exact_sum,
                         "measured_e2e_ms": e2e,
                         "e2e_minus_exact_local_ms": None if e2e is None else e2e - exact_sum,
                         "e2e_over_exact_local": None if e2e is None else e2e / exact_sum,
                         **{f"count_{method}": counts[method] for method in METHOD_TO_KERNEL}})
    with (report / "policy_speed_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(policies[0])); writer.writeheader(); writer.writerows(policies)
    lines = ["# Llama3 prefill speed decomposition", "", "## Per-policy decomposition", "", "| policy | predicted local (ms) | exact local (ms) | local error | E2E (ms) | E2E - exact local | E2E/exact |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in policies:
        e2e = row["measured_e2e_ms"]
        lines.append(f"| {row['policy_id']} | {row['predicted_local_sum_ms']:.2f} | {row['exact_local_sum_ms']:.2f} | {row['local_prediction_error_pct']:.1f}% | " + ("— | — | — |" if e2e is None else f"{e2e:.2f} | {row['e2e_minus_exact_local_ms']:.2f} | {row['e2e_over_exact_local']:.3f} |"))
    lines += ["", "Interpretation: predictor-vs-exact local error diagnoses the kernel model; exact-local-vs-E2E residual diagnoses model/runtime composition."]
    (report / "summary.md").write_text("\n".join(lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
