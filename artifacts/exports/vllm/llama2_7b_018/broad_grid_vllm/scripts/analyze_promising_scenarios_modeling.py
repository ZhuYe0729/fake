#!/usr/bin/env python3
"""Select promising broad-grid scenarios and analyze them with kernel models."""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BROAD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BROAD_ROOT.parents[4]
MODELING_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
if str(MODELING_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELING_ROOT))

from modeling.kernel_predictor import KernelLatencyPredictor  # noqa: E402


METHODS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]
SPEEDUP_METHODS = ["dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4", "hetero"]
LAYERS = 32
LINEARS = [
    ("qkv", 12288, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up", 22016, 4096),
    ("down", 4096, 11008),
]


@dataclass(frozen=True)
class Scenario:
    batch: int
    input_seq: int
    output_seq: int
    measured_best_method: str
    measured_best_speedup: float
    measured_hetero_speedup: str

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.batch, self.input_seq, self.output_seq)

    @property
    def name(self) -> str:
        return f"b{self.batch}_in{self.input_seq}_out{self.output_seq}"


def read_speedups(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def row_to_scenario(row: dict[str, str]) -> Scenario | None:
    if row.get("dense_bf16") != "1.000":
        return None
    values = [(parse_float(row.get(method, "")), method) for method in SPEEDUP_METHODS]
    values = [(value, method) for value, method in values if value is not None]
    if not values:
        return None
    best_speedup, best_method = max(values)
    return Scenario(
        batch=int(row["batch"]),
        input_seq=int(row["input_seq"]),
        output_seq=int(row["output_seq"]),
        measured_best_method=best_method,
        measured_best_speedup=best_speedup,
        measured_hetero_speedup=row.get("hetero", ""),
    )


def select_scenarios(rows: list[dict[str, str]]) -> list[Scenario]:
    candidates = [scenario for row in rows if (scenario := row_to_scenario(row))]
    candidates = [scenario for scenario in candidates if scenario.measured_best_speedup >= 1.70]

    selected: list[Scenario] = []

    def add(items: list[Scenario], limit: int) -> None:
        for item in items:
            if item.key not in {scenario.key for scenario in selected}:
                selected.append(item)
            if len([scenario for scenario in selected if scenario in items]) >= limit:
                break

    long_decode = [
        scenario
        for scenario in candidates
        if scenario.input_seq >= 16384 and scenario.output_seq in {64, 128} and scenario.batch <= 8
    ]
    add(sorted(long_decode, key=lambda item: item.measured_best_speedup, reverse=True), 6)

    prefill_only = [scenario for scenario in candidates if scenario.output_seq == 1]
    add(sorted(prefill_only, key=lambda item: item.measured_best_speedup, reverse=True), 6)

    medium_decode = [
        scenario
        for scenario in candidates
        if scenario.output_seq in {16, 64} and scenario.input_seq < 16384
    ]
    add(sorted(medium_decode, key=lambda item: item.measured_best_speedup, reverse=True), 4)

    return selected


def predict_call(
    predictor: KernelLatencyPredictor,
    method: str,
    m: int,
    n: int,
    k: int,
) -> tuple[float | None, str]:
    selection = predictor.predict(m=m, n=n, k=k)
    by_kernel = {candidate.kernel: candidate for candidate in selection.candidates}
    candidate = by_kernel.get(method)
    if candidate is None:
        return None, "missing"
    if not candidate.supported or candidate.latency_ms is None:
        return None, candidate.reason or "unsupported"
    return float(candidate.latency_ms), candidate.source


def predict_workload_method(
    predictor: KernelLatencyPredictor,
    scenario: Scenario,
    method: str,
) -> tuple[float | None, str]:
    total = 0.0
    reasons: list[str] = []
    prefill_m = scenario.batch * scenario.input_seq
    decode_m = scenario.batch
    for _name, n, k in LINEARS:
        latency, reason = predict_call(predictor, method, prefill_m, n, k)
        if latency is None:
            reasons.append(f"prefill:{reason}")
        else:
            total += latency * LAYERS
        if scenario.output_seq > 0:
            latency, reason = predict_call(predictor, method, decode_m, n, k)
            if latency is None:
                reasons.append(f"decode:{reason}")
            else:
                total += latency * LAYERS * scenario.output_seq
    if reasons:
        return None, "; ".join(sorted(set(reasons)))[:180]
    return total, "OK"


def best_for_call(
    predictor: KernelLatencyPredictor,
    m: int,
    n: int,
    k: int,
) -> tuple[str, float, dict[str, float]]:
    selection = predictor.predict(m=m, n=n, k=k)
    latencies = {
        candidate.kernel: float(candidate.latency_ms)
        for candidate in selection.candidates
        if candidate.supported and candidate.latency_ms is not None
    }
    if not latencies:
        raise RuntimeError(f"no supported kernel for {(m, n, k)}")
    best = min(latencies, key=latencies.get)
    return best, latencies[best], latencies


def predict_workload_best_mixed(
    predictor: KernelLatencyPredictor,
    scenario: Scenario,
) -> tuple[float, Counter[str], list[dict[str, Any]]]:
    total = 0.0
    choices: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    phases = [("prefill", scenario.batch * scenario.input_seq, 1)]
    if scenario.output_seq > 0:
        phases.append(("decode", scenario.batch, scenario.output_seq))
    for phase, m, repeats in phases:
        for linear_name, n, k in LINEARS:
            best, latency, latencies = best_for_call(predictor, m, n, k)
            weighted = latency * LAYERS * repeats
            total += weighted
            choices[best] += 1
            details.append(
                {
                    "phase": phase,
                    "linear": linear_name,
                    "m": m,
                    "n": n,
                    "k": k,
                    "repeats": repeats,
                    "best_kernel": best,
                    "single_call_ms": latency,
                    "weighted_ms": weighted,
                    **{f"{method}_single_call_ms": latencies.get(method, "") for method in METHODS},
                }
            )
    return total, choices, details


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def write_markdown(path: Path, selected_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Promising Scenarios and Kernel-Model Analysis",
        "",
        "本文件从 broad-grid vLLM 实测结果中筛选“存在明显加速空间”的场景，并用 "
        "`fake/kernels/cutlass/cutlass_wrapper/modeling` 的 `KernelLatencyPredictor` "
        "分析 fused Llama2-7B Linear 部分的速度。",
        "",
        "## 筛选标准",
        "",
        "- 只从 `dense_bf16` 为 OK 的可比场景中筛选。",
        "- 至少一个压缩/异构方案的 vLLM 端到端 speedup `>= 1.70x`。",
        "- 保留三类场景：长上下文生成、prefill-only/近似 prefill-only、以及中等输出长度场景。",
        "- 这些场景不要求当前 hetero 实测已经最好；目标是找到压缩方案整体有明显收益、层异构优化值得发挥的区域。",
        "",
        "## 预测假设",
        "",
        "- 只预测 Linear kernel latency，不包含 attention、KV cache、scheduler、sampling 和 vLLM runtime overhead。",
        "- 按 vLLM/Llama fused Linear 近似：`qkv(12288,4096)`、`o_proj(4096,4096)`、"
        "`gate_up(22016,4096)`、`down(4096,11008)`，每层 4 个 fused Linear，共 32 层。",
        "- prefill 使用 `m=batch*input_seq`；decode 使用 `m=batch` 并重复 `output_seq` 次。",
        "- `best_mixed` 是逐 phase/Linear shape 选择预测 latency 最低且满足 kernel 支持约束的 kernel；"
        "它是速度上界分析，不包含精度约束。",
        "- 主表中空白的 single-method latency 表示该方法在 modeling 约束下无法完整覆盖该场景的 prefill+decode Linear 调用；"
        "具体原因见 CSV 的 `pred_*_status` 列。",
        "",
        "## Selected Scenario Summary",
        "",
        "| scenario | measured_best | measured_best_speedup | measured_hetero_speedup | "
        "pred_dense_bf16_ms | pred_dense_nvfp4_ms | pred_sparse_bf16_ms | pred_sparse_nvfp4_ms | "
        "pred_marlin_nvfp4_ms | pred_best_single | pred_best_mixed_ms | pred_best_mixed_speedup | "
        "mixed_vs_best_single | best_mixed_choices |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in selected_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(row["measured_best_method"]),
                    str(row["measured_best_speedup"]),
                    str(row["measured_hetero_speedup"]),
                    str(row["pred_dense_bf16_ms"]),
                    str(row["pred_dense_nvfp4_ms"]),
                    str(row["pred_sparse_bf16_ms"]),
                    str(row["pred_sparse_nvfp4_ms"]),
                    str(row["pred_marlin_nvfp4_ms"]),
                    f"{row['pred_best_single_method']}:{row['pred_best_single_ms']}",
                    str(row["pred_best_mixed_ms"]),
                    str(row["pred_best_mixed_speedup"]),
                    str(row["pred_best_mixed_vs_best_single"]),
                    str(row["best_mixed_choices"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- vLLM 实测中最明显的加速集中在长上下文、小到中等 batch、较长输出，以及 output=1 的 prefill-dominant 场景。",
            "- kernel 模型显示 best_mixed 通常会在 prefill 的大 `m` Linear 上偏向 sparse/dense NVFP4 类 kernel，"
            "在 decode 的小 `m` Linear 上偏向 marlin 或 dense BF16/NVFP4，避免单一方法在某些 phase 不占优。",
            "- 若后续要把 best_mixed 变成真实策略，需要再叠加精度约束、fused module 约束以及 vLLM backend 的实际可用 kernel 约束。",
            "",
            "## Files",
            "",
            "- `promising_scenarios_modeling.csv`: 场景级汇总。",
            "- `promising_scenarios_modeling_details.csv`: 每个场景、phase、fused Linear 的逐 kernel 预测和 best choice。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    speedup_rows = read_speedups(BROAD_ROOT / "summary/broad_grid_speedup_table.csv")
    scenarios = select_scenarios(speedup_rows)
    predictor = KernelLatencyPredictor()

    selected_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        method_totals: dict[str, float | None] = {}
        method_status: dict[str, str] = {}
        for method in METHODS:
            total, status = predict_workload_method(predictor, scenario, method)
            method_totals[method] = total
            method_status[method] = status

        mixed_total, choices, details = predict_workload_best_mixed(predictor, scenario)
        dense_total = method_totals["dense_bf16"]
        mixed_speedup = None if dense_total in (None, 0) else float(dense_total) / mixed_total
        supported_single = {
            method: total
            for method, total in method_totals.items()
            if total is not None and total > 0
        }
        best_single_method = min(supported_single, key=supported_single.get)
        best_single_total = supported_single[best_single_method]
        mixed_vs_best_single = best_single_total / mixed_total
        selected_rows.append(
            {
                "scenario": scenario.name,
                "batch": scenario.batch,
                "input_seq": scenario.input_seq,
                "output_seq": scenario.output_seq,
                "measured_best_method": scenario.measured_best_method,
                "measured_best_speedup": f"{scenario.measured_best_speedup:.3f}",
                "measured_hetero_speedup": scenario.measured_hetero_speedup,
                **{f"pred_{method}_ms": fmt(method_totals[method]) for method in METHODS},
                **{f"pred_{method}_status": method_status[method] for method in METHODS},
                "pred_best_single_method": best_single_method,
                "pred_best_single_ms": fmt(best_single_total),
                "pred_best_mixed_ms": fmt(mixed_total),
                "pred_best_mixed_speedup": fmt(mixed_speedup),
                "pred_best_mixed_vs_best_single": fmt(mixed_vs_best_single),
                "best_mixed_choices": ",".join(f"{kernel}:{count}" for kernel, count in sorted(choices.items())),
            }
        )
        for detail in details:
            detail_rows.append({"scenario": scenario.name, **detail})

    out_dir = BROAD_ROOT / "summary"
    write_csv(out_dir / "promising_scenarios_modeling.csv", selected_rows)
    write_csv(out_dir / "promising_scenarios_modeling_details.csv", detail_rows)
    write_markdown(out_dir / "promising_scenarios_modeling.md", selected_rows, detail_rows)


if __name__ == "__main__":
    main()
