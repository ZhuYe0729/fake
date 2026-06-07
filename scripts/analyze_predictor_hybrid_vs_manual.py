#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
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
    policy_to_dict,
    select_offline_hybrid_policy,
    write_policy_csv,
)
from modeling.kernel_predictor import KernelLatencyPredictor  # noqa: E402


OUT_DIR = REPO_ROOT / "artifacts/results/benchmarks/hybrid/pred"
MANUAL_DIR = REPO_ROOT / "artifacts/results/benchmarks/hybrid/manual"
PREFILL_KERNEL_DATA = MANUAL_DIR / "prefill_only/kernel_data"
QWEN_KERNEL_CSV = REPO_ROOT / "artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_curves.csv"

KERNELS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]
MODELS = ["Llama-2-7B", "Llama-3.1-8B", "Qwen3.5-9B"]
SCENARIOS = {
    "prefill_only": ScenarioSpec(batch_size=16, input_tokens=1024, output_tokens=0),
    "normal_01": ScenarioSpec(batch_size=1, input_tokens=16384, output_tokens=32),
}

MANUAL_NORMAL = {
    "Llama-2-7B": {
        "dense_bf16": (2438.0, 1.00),
        "dense_nvfp4": (3249.0, 0.75),
        "sparse_bf16": (2295.0, 1.06),
        "sparse_nvfp4": (3491.0, 0.70),
        "marlin_nvfp4": (2322.0, 1.05),
        "hybrid": (1930.0, 1.26),
    },
    "Llama-3.1-8B": {
        "dense_bf16": (2270.0, 1.00),
        "dense_nvfp4": (2989.0, 0.76),
        "sparse_bf16": (2331.0, 0.97),
        "sparse_nvfp4": (3283.0, 0.69),
        "marlin_nvfp4": (2308.0, 0.98),
        "hybrid": (2002.0, 1.13),
    },
    "Qwen3.5-9B": {
        "dense_bf16": (4190.0, 1.00),
        "dense_nvfp4": (5133.0, 0.82),
        "sparse_bf16": (3308.0, 1.27),
        "sparse_nvfp4": (4905.0, 0.85),
        "marlin_nvfp4": (4357.0, 0.96),
        "hybrid": (3308.0, 1.27),
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor = KernelLatencyPredictor()
    summary_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []

    for model in MODELS:
        linears = load_model_linears(model)
        for scenario_name, scenario in SCENARIOS.items():
            policy = select_offline_hybrid_policy(linears, scenario, predictor)
            policy_path = OUT_DIR / f"{model_key(model)}_{scenario_name}_policy.json"
            policy_path.write_text(json.dumps(policy_to_dict(policy), indent=2) + "\n")
            write_policy_csv(policy, OUT_DIR / f"{model_key(model)}_{scenario_name}_policy.csv")

            manual_policy = manual_hybrid_policy(model, scenario_name, linears)
            for module in policy.modules:
                manual_prefill, manual_decode = manual_policy.get(module.name, ("", ""))
                pred_prefill = module.selected_prefill_backend or ""
                pred_decode = module.selected_decode_backend or ""
                strategy_rows.append(
                    {
                        "model": model,
                        "scenario": scenario_name,
                        "linear_group": module.name,
                        "count": module.count,
                        "n": module.n,
                        "k": module.k,
                        "manual_prefill_backend": manual_prefill,
                        "manual_decode_backend": manual_decode,
                        "pred_prefill_backend": pred_prefill,
                        "pred_decode_backend": pred_decode,
                        "same_strategy": (manual_prefill, manual_decode) == (pred_prefill, pred_decode),
                        "pred_total_ms": module.selected_total_ms,
                        "pred_conversion_ms": module.selected_conversion_ms,
                    }
                )

            diff_counter = Counter(
                (
                    row["manual_prefill_backend"],
                    row["manual_decode_backend"],
                    row["pred_prefill_backend"],
                    row["pred_decode_backend"],
                )
                for row in strategy_rows
                if row["model"] == model and row["scenario"] == scenario_name
            )
            for (m_pf, m_dec, p_pf, p_dec), groups in sorted(diff_counter.items()):
                layer_count = sum(
                    row["count"]
                    for row in strategy_rows
                    if row["model"] == model
                    and row["scenario"] == scenario_name
                    and row["manual_prefill_backend"] == m_pf
                    and row["manual_decode_backend"] == m_dec
                    and row["pred_prefill_backend"] == p_pf
                    and row["pred_decode_backend"] == p_dec
                )
                diff_rows.append(
                    {
                        "model": model,
                        "scenario": scenario_name,
                        "manual_prefill_backend": m_pf,
                        "manual_decode_backend": m_dec,
                        "pred_prefill_backend": p_pf,
                        "pred_decode_backend": p_dec,
                        "linear_groups": groups,
                        "linear_layers": layer_count,
                    }
                )

            pred_methods = predicted_method_totals(linears, scenario, predictor)
            pred_hybrid = sum(float(module.selected_total_ms or 0.0) for module in policy.modules)
            pred_methods["predictor_hybrid"] = pred_hybrid
            pred_dense = pred_methods["dense_bf16"]
            manual = manual_results(model, scenario_name)
            for method, pred_ms in pred_methods.items():
                manual_method = "hybrid" if method == "predictor_hybrid" else method
                manual_ms, manual_speedup = manual.get(manual_method, (None, None))
                summary_rows.append(
                    {
                        "model": model,
                        "scenario": scenario_name,
                        "batch_size": scenario.batch_size,
                        "input_tokens": scenario.input_tokens,
                        "output_tokens": scenario.output_tokens,
                        "m_prefill": scenario.m_prefill,
                        "m_decode": scenario.m_decode,
                        "method": method,
                        "pred_linear_e2e_ms": f"{pred_ms:.4f}",
                        "pred_speedup_vs_pred_dense_bf16": f"{pred_dense / pred_ms:.4f}" if pred_ms > 0 else "",
                        "manual_e2e_or_prefill_ms": "" if manual_ms is None else f"{manual_ms:.4f}",
                        "manual_speedup_vs_dense_bf16": "" if manual_speedup is None else f"{manual_speedup:.4f}",
                    }
                )

    write_csv(OUT_DIR / "predictor_vs_manual_summary.csv", summary_rows)
    write_csv(OUT_DIR / "strategy_comparison.csv", strategy_rows)
    write_csv(OUT_DIR / "strategy_diff_summary.csv", diff_rows)
    write_summary_md(summary_rows, diff_rows)
    print(f"wrote predictor hybrid comparison to {OUT_DIR}")


def load_model_linears(model: str) -> list[LinearShapeSpec]:
    if model == "Llama-2-7B":
        return linears_from_kernel_csv(PREFILL_KERNEL_DATA / "llama_2_7b_kernel_m16384.csv")
    if model == "Llama-3.1-8B":
        return linears_from_kernel_csv(PREFILL_KERNEL_DATA / "llama_3.1_8b_kernel_m16384.csv")
    if model == "Qwen3.5-9B":
        return linears_from_kernel_csv(QWEN_KERNEL_CSV)
    raise ValueError(f"unknown model: {model}")


def linears_from_kernel_csv(path: Path) -> list[LinearShapeSpec]:
    seen: dict[str, LinearShapeSpec] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            group = row["linear_group"]
            if group in seen:
                continue
            seen[group] = LinearShapeSpec(
                name=group,
                n=int(row["n"]),
                k=int(row["k"]),
                count=int(row["linear_count"]),
            )
    return list(seen.values())


def predicted_method_totals(
    linears: list[LinearShapeSpec],
    scenario: ScenarioSpec,
    predictor: KernelLatencyPredictor,
) -> dict[str, float]:
    totals = {kernel: 0.0 for kernel in KERNELS}
    for linear in linears:
        prefill = {c.kernel: c for c in predictor.predict(scenario.m_prefill, linear.n, linear.k).candidates}
        decode = {c.kernel: c for c in predictor.predict(scenario.m_decode, linear.n, linear.k).candidates}
        conversions = {c.conversion: c for c in predictor.predict_conversion(linear.n, linear.k)}
        for kernel in KERNELS:
            pf = prefill[kernel]
            dec = decode[kernel]
            decode_needed = scenario.output_tokens != 0
            if not pf.supported or pf.latency_ms is None or (decode_needed and (not dec.supported or dec.latency_ms is None)):
                totals[kernel] += float("inf")
                continue
            conversion_ms = conversion_cost_for_kernel(kernel, conversions)
            if conversion_ms is None:
                totals[kernel] += float("inf")
                continue
            decode_ms = 0.0 if not decode_needed else float(dec.latency_ms)
            totals[kernel] += linear.count * (float(pf.latency_ms) + scenario.output_tokens * decode_ms + conversion_ms)
    return totals


def conversion_cost_for_kernel(kernel: str, conversions: dict[str, Any]) -> float | None:
    conversion_name = {
        "dense_nvfp4": "canonical_to_cutlass",
        "marlin_nvfp4": "canonical_to_marlin",
    }.get(kernel)
    if conversion_name is None:
        return 0.0
    candidate = conversions.get(conversion_name)
    if candidate is None or not candidate.supported or candidate.latency_ms is None:
        return None
    return float(candidate.latency_ms)


def manual_hybrid_policy(
    model: str,
    scenario_name: str,
    linears: list[LinearShapeSpec],
) -> dict[str, tuple[str, str]]:
    if scenario_name == "prefill_only":
        return manual_prefill_policy(model)
    if model in ("Llama-2-7B", "Llama-3.1-8B"):
        return {linear.name: ("dense_nvfp4", "marlin_nvfp4") for linear in linears}
    if model == "Qwen3.5-9B":
        out = {}
        for linear in linears:
            if linear.n == 32 or linear.name.endswith(("self_attn.k_proj", "self_attn.v_proj")):
                out[linear.name] = ("dense_bf16", "dense_bf16")
            else:
                out[linear.name] = ("sparse_bf16", "sparse_bf16")
        return out
    raise ValueError(f"unknown model: {model}")


def manual_prefill_policy(model: str) -> dict[str, tuple[str, str]]:
    rows = []
    if model == "Llama-2-7B":
        path = PREFILL_KERNEL_DATA / "llama_2_7b_kernel_m16384.csv"
    elif model == "Llama-3.1-8B":
        path = PREFILL_KERNEL_DATA / "llama_3.1_8b_kernel_m16384.csv"
    elif model == "Qwen3.5-9B":
        path = QWEN_KERNEL_CSV
    else:
        raise ValueError(f"unknown model: {model}")
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(row["m"]) == 16384 and row["status"] == "pass":
                row["latency_ms"] = float(row["latency_ms"])
                rows.append(row)
    best: dict[str, tuple[str, float]] = {}
    for row in rows:
        group = row["linear_group"]
        current = best.get(group)
        if current is None or row["latency_ms"] < current[1]:
            best[group] = (row["kernel"], row["latency_ms"])
    return {group: (kernel, kernel) for group, (kernel, _lat) in best.items()}


def manual_results(model: str, scenario_name: str) -> dict[str, tuple[float | None, float | None]]:
    if scenario_name == "normal_01":
        return MANUAL_NORMAL[model]
    path = {
        "Llama-2-7B": MANUAL_DIR / "prefill_only/prefill_hybrid_llama_2_7b.csv",
        "Llama-3.1-8B": MANUAL_DIR / "prefill_only/prefill_hybrid_llama_31_8b.csv",
        "Qwen3.5-9B": MANUAL_DIR / "prefill_only/prefill_hybrid_qwen35_9b.csv",
    }[model]
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["method"]] = (float(row["prefill_ms"]), float(row["speedup_vs_dense_bf16"]))
    return out


def model_key(model: str) -> str:
    return model.lower().replace(".", "_").replace("-", "_")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(summary_rows: list[dict[str, Any]], diff_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Predictor Hybrid vs Manual Hybrid",
        "",
        "## Scenarios",
        "- `prefill_only`: batch_size=16, input_tokens=1024, output_tokens=0.",
        "- `normal_01`: batch_size=1, input_tokens=16384, output_tokens=32.",
        "",
        "Predicted latency is summed over compressible Linear layers using `KernelLatencyPredictor`; manual columns come from existing benchmark artifacts.",
        "",
        "## Predictor Hybrid Summary",
        "",
        "| Model | Scenario | Pred hybrid linear ms | Pred speedup | Manual hybrid ms | Manual speedup |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["method"] != "predictor_hybrid":
            continue
        lines.append(
            "| {model} | {scenario} | {pred_linear_e2e_ms} | {pred_speedup_vs_pred_dense_bf16}x | "
            "{manual_e2e_or_prefill_ms} | {manual_speedup_vs_dense_bf16}x |".format(**row)
        )
    lines.extend(["", "## Strategy Difference Summary", "", "| Model | Scenario | Manual | Predictor | Groups | Layers |", "|---|---|---|---|---:|---:|"])
    for row in diff_rows:
        manual = f"{row['manual_prefill_backend']}->{row['manual_decode_backend']}"
        pred = f"{row['pred_prefill_backend']}->{row['pred_decode_backend']}"
        lines.append(
            f"| {row['model']} | {row['scenario']} | {manual} | {pred} | "
            f"{row['linear_groups']} | {row['linear_layers']} |"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
