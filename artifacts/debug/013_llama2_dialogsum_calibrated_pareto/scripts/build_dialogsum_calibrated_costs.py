#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
SOURCE_010 = FAKE_ROOT / "artifacts/debug/010_llama2_normal02_pareto_handoff"
SOURCE_012 = FAKE_ROOT / "artifacts/debug/012_llama2_dialogsum_pareto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DialogSum-calibrated per-module candidate costs.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-010", type=Path, default=SOURCE_010)
    parser.add_argument("--source-012", type=Path, default=SOURCE_012)
    parser.add_argument("--rouge-weight", type=float, default=100.0)
    parser.add_argument("--quality-floor", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_candidates = read_csv(args.source_010 / "costs" / "module_method_candidates.csv")
    dialogsum = read_csv(args.source_012 / "summary" / "dialogsum_pareto" / "dialogsum_pareto_summary.csv")
    method_losses = empirical_method_losses(dialogsum, rouge_weight=args.rouge_weight)
    original_totals = original_method_quality_totals(source_candidates)
    scales = {}
    for method, loss in method_losses.items():
        if method == "dense_bf16":
            scales[method] = 0.0
            continue
        total = original_totals.get(method, 0.0)
        scales[method] = loss / total if total > 0 else 0.0

    rows = []
    for row in source_candidates:
        item = dict(row)
        method = item["method"]
        original_quality = f(item, "quality_cost")
        calibrated = 0.0 if method == "dense_bf16" else max(0.0, original_quality * scales.get(method, 0.0))
        if calibrated > 0.0:
            calibrated = max(calibrated, args.quality_floor)
        item["original_quality_cost"] = original_quality
        item["dialogsum_method_loss"] = method_losses.get(method, "")
        item["dialogsum_quality_scale"] = scales.get(method, "")
        item["quality_cost"] = calibrated
        item["quality_formula"] = f"dialogsum_uniform_calibrated_rouge_weight_{args.rouge_weight:g}"
        rows.append(item)

    out_csv = args.output_root / "costs" / "module_method_candidates.csv"
    write_csv(out_csv, rows)
    write_json(
        args.output_root / "costs" / "dialogsum_calibration_metadata.json",
        {
            "source_candidates": str(args.source_010 / "costs" / "module_method_candidates.csv"),
            "dialogsum_summary": str(args.source_012 / "summary" / "dialogsum_pareto" / "dialogsum_pareto_summary.csv"),
            "rouge_weight": args.rouge_weight,
            "method_losses": method_losses,
            "original_quality_totals": original_totals,
            "method_scales": scales,
            "notes": [
                "loss = max(0, method_nll - dense_bf16_nll) + rouge_weight * max(0, dense_bf16_rougeL - method_rougeL)",
                "per-module relative sensitivity comes from the original local error proxy; method-level scale comes from full DialogSum uniform results.",
            ],
        },
    )
    print(f"wrote {len(rows)} calibrated rows to {out_csv}")


def empirical_method_losses(rows: list[dict[str, Any]], *, rouge_weight: float) -> dict[str, float]:
    uniforms = {row["label"]: row for row in rows if row.get("kind") == "uniform"}
    dense = uniforms["dense_bf16"]
    dense_nll = f(dense, "conditional_nll")
    dense_rouge = f(dense, "rougeL")
    losses = {}
    for method, row in uniforms.items():
        nll_loss = max(0.0, f(row, "conditional_nll") - dense_nll)
        rouge_loss = max(0.0, dense_rouge - f(row, "rougeL"))
        losses[method] = nll_loss + rouge_weight * rouge_loss
    return losses


def original_method_quality_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        if str(row.get("supported", "True")).lower() != "true":
            continue
        method = row["method"]
        totals[method] = totals.get(method, 0.0) + f(row, "quality_cost")
    return totals


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
