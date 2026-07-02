#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("artifacts/debug/028_fakevlm_linear_time_proportion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FakeVLM linear time proportion results.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.output_root / "results" / "fakevlm_linear_proportion_raw.csv"
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(raw_path)
    summary_rows = build_summary(rows)
    write_csv(summary_dir / "fakevlm_linear_proportion_summary.csv", summary_rows)
    write_report(summary_dir / "analysis_report.md", summary_rows)
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Report: {summary_dir / 'analysis_report.md'}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda item: item.get("workload", "")):
        out.append(
            {
                "workload": row.get("workload", ""),
                "batch_size": row.get("batch_size", ""),
                "input_tokens": row.get("input_tokens", ""),
                "output_tokens": row.get("output_tokens", ""),
                "prefill_ms": row.get("prefill_ms", ""),
                "decode_avg_ms": row.get("decode_avg_ms", ""),
                "e2e_ms": row.get("e2e_ms", ""),
                "prefill_all_linear_pct": fmt(row.get("prefill_all_linear_pct")),
                "prefill_language_linear_pct": fmt(row.get("prefill_language_linear_pct")),
                "prefill_vision_linear_pct": fmt(row.get("prefill_vision_linear_pct")),
                "prefill_projector_linear_pct": fmt(row.get("prefill_projector_linear_pct")),
                "decode_all_linear_pct": fmt(row.get("decode_all_linear_pct")),
                "decode_language_linear_pct": fmt(row.get("decode_language_linear_pct")),
                "decode_vision_linear_pct": fmt(row.get("decode_vision_linear_pct")),
                "decode_projector_linear_pct": fmt(row.get("decode_projector_linear_pct")),
                "all_linear_count": row.get("all_linear_count", ""),
                "language_linear_count": row.get("language_linear_count", ""),
                "vision_linear_count": row.get("vision_linear_count", ""),
                "projector_linear_count": row.get("projector_linear_count", ""),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("# FakeVLM Linear 时间占比分析报告")
    lines.append("")
    lines.append("## 数据概况")
    lines.append("")
    lines.append(f"- Workload 数: {len(rows)}")
    if rows:
        first = rows[0]
        lines.append(
            f"- Linear 模块数: all={first.get('all_linear_count')}, "
            f"language={first.get('language_linear_count')}, vision={first.get('vision_linear_count')}, "
            f"projector={first.get('projector_linear_count')}"
        )
    lines.append("")
    if not rows:
        lines.append("没有可用结果。")
        path.write_text("\n".join(lines))
        return

    lines.append("## 核心结果")
    lines.append("")
    lines.append("| Workload | Batch | 输入 | 输出 | Prefill Linear% | Language% | Vision% | Projector% | Decode Linear% | Decode Language% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['workload']} | {row['batch_size']} | {row['input_tokens']} | {row['output_tokens']} | "
            f"{row['prefill_all_linear_pct']}% | {row['prefill_language_linear_pct']}% | "
            f"{row['prefill_vision_linear_pct']}% | {row['prefill_projector_linear_pct']}% | "
            f"{row['decode_all_linear_pct']}% | {row['decode_language_linear_pct']}% |"
        )
    lines.append("")
    lines.append("## 解读要点")
    lines.append("")
    lines.append("- Prefill 包含 vision tower、multimodal projector 和 language model，因此总 linear 占比应同时看 language/vision/projector 拆分。")
    lines.append("- Decode 使用 KV cache 后通常只走 language model 路径，vision/projector linear 占比应接近 0；如果不为 0，需要优先检查模型 forward 是否重复传入图像特征。")
    lines.append("- 和纯 LLM 对比时，FakeVLM prefill denominator 更大，因为图像编码和多模态投影也在完整 forward 内。")
    lines.append("- 该报告使用 CUDA event hook 计时，适合分析比例趋势；严格 kernel attribution 仍应使用 nsys 交叉验证。")
    path.write_text("\n".join(lines))


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None, "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    parsed = as_float(value)
    return "N/A" if parsed is None else f"{parsed:.1f}"


if __name__ == "__main__":
    main()
