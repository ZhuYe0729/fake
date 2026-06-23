#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_DIR = REPO_ROOT / "artifacts/debug/022_linear_time_proportion_study/llama2_7b"
QWEN_022_DIR = REPO_ROOT / "artifacts/debug/022_linear_time_proportion_study"
SUMMARY_DIR = ARTIFACT_DIR / "summary"


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    speed_rows = load_many(ARTIFACT_DIR / "speed", "llama2_7b_speed_shard*.csv")
    breakdown_rows = load_many(ARTIFACT_DIR / "breakdown_coarse", "llama2_7b_breakdown_coarse_shard*.csv")
    print(f"Loaded speed rows: {len(speed_rows)}")
    print(f"Loaded breakdown rows: {len(breakdown_rows)}")

    summary_rows = build_summary(speed_rows, breakdown_rows)
    write_csv(SUMMARY_DIR / "llama2_linear_proportion_summary.csv", summary_rows)
    write_report(summary_rows)
    write_qwen_context(summary_rows)
    print(f"Summary written: {SUMMARY_DIR}")


def load_many(directory: Path, pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob(pattern)):
        with path.open() as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def build_summary(speed_rows: list[dict[str, str]], breakdown_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    breakdown_lookup = {
        key_for(row): row
        for row in breakdown_rows
        if row.get("status", "OK") == "OK"
    }
    out: list[dict[str, Any]] = []
    for speed in sorted(speed_rows, key=sort_key):
        bd = breakdown_lookup.get(key_for(speed), {})
        status = speed.get("status", "")
        row = {
            "model": "llama2-7b",
            "batch_size": speed.get("batch_size", ""),
            "input_tokens": speed.get("input_tokens", ""),
            "output_tokens": speed.get("output_tokens", ""),
            "scenario": "prefill_only" if int_or_zero(speed.get("output_tokens")) <= 1 else "prefill_decode",
            "prefill_ms": speed.get("prefill_ms", ""),
            "decode_total_ms": speed.get("decode_total_ms", ""),
            "decode_per_token_ms": speed.get("decode_per_token_ms", ""),
            "first_decode_ms": speed.get("first_decode_ms", ""),
            "prefill_all_linear_pct": fmt_pct(bd.get("prefill_all_linear_pct")),
            "decode_all_linear_pct": fmt_pct(bd.get("decode_all_linear_pct")),
            "prefill_self_attn_block_pct": fmt_pct(bd.get("prefill_self_attn_block_pct")),
            "decode_self_attn_block_pct": fmt_pct(bd.get("decode_self_attn_block_pct")),
            "prefill_mlp_block_pct": fmt_pct(bd.get("prefill_mlp_block_pct")),
            "decode_mlp_block_pct": fmt_pct(bd.get("decode_mlp_block_pct")),
            "prefill_norm_pct": fmt_pct(bd.get("prefill_norm_pct")),
            "decode_norm_pct": fmt_pct(bd.get("decode_norm_pct")),
            "prefill_lm_head_pct": fmt_pct(bd.get("prefill_lm_head_pct")),
            "decode_lm_head_pct": fmt_pct(bd.get("decode_lm_head_pct")),
            "prefill_other_pct": fmt_pct(bd.get("prefill_other_pct")),
            "decode_other_pct": fmt_pct(bd.get("decode_other_pct")),
            "status": status or "OK",
            "error_msg": speed.get("error_msg", ""),
        }
        out.append(row)
    return out


def key_for(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("batch_size", ""), row.get("input_tokens", ""), row.get("output_tokens", ""))


def sort_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (
        int_or_zero(row.get("batch_size")),
        int_or_zero(row.get("input_tokens")),
        int_or_zero(row.get("output_tokens")),
    )


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_or_none(value: Any) -> float | None:
    try:
        if value in ("", None, "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_pct(value: Any) -> str:
    parsed = float_or_none(value)
    return "N/A" if parsed is None else f"{parsed:.1f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def valid_breakdown_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "OK" and row.get("prefill_all_linear_pct") != "N/A"]


def average(rows: list[dict[str, Any]], field: str) -> float:
    values = [float_or_none(row.get(field)) for row in rows]
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def group_average(rows: list[dict[str, Any]], group_field: str, value_field: str) -> list[tuple[str, float, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, ""))].append(row)
    out = []
    for key in sorted(grouped, key=lambda x: int_or_zero(x)):
        out.append((key, average(grouped[key], value_field), len(grouped[key])))
    return out


def write_report(rows: list[dict[str, Any]]) -> None:
    valid = valid_breakdown_rows(rows)
    ok_speed = [row for row in rows if row.get("status") == "OK"]
    oom_rows = [row for row in rows if row.get("status") == "OOM"]

    lines: list[str] = []
    lines.append("# Llama2-7B Linear 时间占比分析报告")
    lines.append("")
    lines.append("## 测试方法")
    lines.append("")
    lines.append("- 模型: Llama2-7B dense BF16")
    lines.append("- Batch size: 1, 4, 16, 32, 64")
    lines.append("- 输入 token 数: 16, 64, 256, 1024, 4096, 8192")
    lines.append("- Speed 输出 token 数: 1, 32, 128, 256")
    lines.append("- Breakdown 输出 token 数: 1, 32")
    lines.append("- 测量方式: CUDA event hook 级别计时，所有 nn.Linear 聚合为 all_linear")
    lines.append("")
    lines.append("## 数据概况")
    lines.append("")
    lines.append(f"- Speed OK 配置数: {len(ok_speed)}")
    lines.append(f"- Breakdown OK 配置数: {len(valid)}")
    lines.append(f"- OOM 配置数: {len(oom_rows)}")
    lines.append("")

    if not valid:
        lines.append("没有可用 breakdown 数据。")
        (SUMMARY_DIR / "analysis_report.md").write_text("\n".join(lines))
        return

    sorted_prefill = sorted(valid, key=lambda row: float_or_none(row["prefill_all_linear_pct"]) or -1.0, reverse=True)
    sorted_decode = sorted(valid, key=lambda row: float_or_none(row["decode_all_linear_pct"]) or -1.0, reverse=True)

    lines.append("## 核心发现")
    lines.append("")
    lines.append("### Prefill Linear 占比最高的配置")
    lines.append("")
    lines.append("| Batch | 输入 | 输出 | Prefill Linear% | Decode Linear% | Prefill ms |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in sorted_prefill[:10]:
        lines.append(
            f"| {row['batch_size']} | {row['input_tokens']} | {row['output_tokens']} | "
            f"{row['prefill_all_linear_pct']}% | {row['decode_all_linear_pct']}% | {row['prefill_ms']} |"
        )
    lines.append("")

    lines.append("### Decode Linear 占比最高的配置")
    lines.append("")
    lines.append("| Batch | 输入 | 输出 | Prefill Linear% | Decode Linear% | First Decode ms |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in sorted_decode[:10]:
        lines.append(
            f"| {row['batch_size']} | {row['input_tokens']} | {row['output_tokens']} | "
            f"{row['prefill_all_linear_pct']}% | {row['decode_all_linear_pct']}% | {row['first_decode_ms']} |"
        )
    lines.append("")

    lines.append("### Batch Size 影响")
    lines.append("")
    lines.append("| Batch | 平均 Prefill Linear% | 平均 Decode Linear% | 样本数 |")
    lines.append("|---:|---:|---:|---:|")
    for batch, avg_prefill, count in group_average(valid, "batch_size", "prefill_all_linear_pct"):
        subset = [row for row in valid if row["batch_size"] == batch]
        lines.append(f"| {batch} | {avg_prefill:.1f}% | {average(subset, 'decode_all_linear_pct'):.1f}% | {count} |")
    lines.append("")

    lines.append("### 输入长度影响")
    lines.append("")
    lines.append("| 输入 Token | 平均 Prefill Linear% | 平均 Decode Linear% | 样本数 |")
    lines.append("|---:|---:|---:|---:|")
    for input_tokens, avg_prefill, count in group_average(valid, "input_tokens", "prefill_all_linear_pct"):
        subset = [row for row in valid if row["input_tokens"] == input_tokens]
        lines.append(f"| {input_tokens} | {avg_prefill:.1f}% | {average(subset, 'decode_all_linear_pct'):.1f}% | {count} |")
    lines.append("")

    lines.append("## 理论解读")
    lines.append("")
    lines.append("- Prefill 阶段的 linear 是形状近似 `[batch * seq, hidden] x [hidden, out]` 的 GEMM；batch 和输入长度增大后，GEMM 更容易吃满 Tensor Core，因此 linear 占比通常上升。")
    lines.append("- 输入过长时，attention core 和 KV/cache/layout 相关开销会随上下文增长，linear 占比可能不再继续上升。")
    lines.append("- Decode 阶段每步只处理一个新 token，linear 形状更接近小 M GEMM/GEMV，实际更容易受访存、kernel launch、KV cache 读取和 attention softmax 影响，因此 linear 时间占比通常显著低于 prefill。")
    lines.append("- Llama2 是标准 full attention 架构，没有 Qwen3.5 的 hybrid linear attention 层；因此和 Qwen 对照时，应重点区分架构差异、hidden/FFN 宽度、attention backend 和上下文长度。")
    lines.append("")
    lines.append("## 压缩策略启示")
    lines.append("")
    lines.append("- 大 batch prefill 是 linear 压缩最有希望带来端到端收益的区域。")
    lines.append("- Decode-heavy 场景不能只看参数量或 FLOPs；如果 measured linear 占比只有 20-30%，Amdahl 上限会很快限制只优化 linear 的收益。")

    (SUMMARY_DIR / "analysis_report.md").write_text("\n".join(lines))


def write_qwen_context(rows: list[dict[str, Any]]) -> None:
    valid = valid_breakdown_rows(rows)
    llama_prefill = average(valid, "prefill_all_linear_pct")
    llama_decode = average(valid, "decode_all_linear_pct")

    qwen_summary = QWEN_022_DIR / "summary/linear_proportion_summary.csv"
    qwen_rows = []
    if qwen_summary.exists():
        with qwen_summary.open() as f:
            qwen_rows = list(csv.DictReader(f))
    qwen_valid = [row for row in qwen_rows if row.get("prefill_all_linear_pct") not in ("", "N/A")]

    lines = []
    lines.append("# Qwen 022 vs Llama2-7B Context")
    lines.append("")
    lines.append("This file is an explanatory context note. It does not modify the existing Qwen 022 summary.")
    lines.append("")
    lines.append("| Model Set | Avg Prefill Linear% | Avg Decode Linear% | Rows |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Llama2-7B | {llama_prefill:.1f}% | {llama_decode:.1f}% | {len(valid)} |")
    if qwen_valid:
        lines.append(
            f"| Existing Qwen 022 | {average(qwen_valid, 'prefill_all_linear_pct'):.1f}% | "
            f"{average(qwen_valid, 'decode_all_linear_pct'):.1f}% | {len(qwen_valid)} |"
        )
    else:
        lines.append("| Existing Qwen 022 | N/A | N/A | 0 |")
    lines.append("")
    lines.append("Interpretation should account for architecture and matrix-shape differences, not just parameter count.")
    (SUMMARY_DIR / "qwen_vs_llama_context.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
