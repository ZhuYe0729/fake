#!/usr/bin/env python3
"""Validate archive structure, provenance hashes, and source-derived paper cells."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from daso.reporting.provenance import validate_manifest
from build_tables import (ablation_values, expected_tables, format2,
                          main_table_values, pro6000_values)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select(path: Path, **where: str) -> dict[str, str]:
    matches = [row for row in read_rows(path)
               if all(row.get(key) == value for key, value in where.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row in {path} for {where}, found {len(matches)}")
    return matches[0]


def table_environment(text: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    end = text.find(marker)
    if end < 0:
        raise ValueError(f"missing table label {label}")
    start = max(text.rfind(r"\begin{table}", 0, end),
                text.rfind(r"\begin{table*}", 0, end))
    if start < 0:
        raise ValueError(f"missing table environment for {label}")
    return text[start:end]


def clean_cell(cell: str) -> str:
    value = cell.strip().replace(r"\\", "")
    wrapper = re.compile(r"\\[A-Za-z]+\{([^{}]*)\}")
    while wrapper.search(value):
        value = wrapper.sub(r"\1", value)
    return value.strip()


def parse_rows(table: str, *, default_model: str | None = None) -> dict[tuple[str, str], list[str]]:
    model = default_model
    output: dict[tuple[str, str], list[str]] = {}
    for line in table.splitlines():
        if "Llama-2" in line:
            model = "llama2"
        elif "Llama-3.1" in line:
            model = "llama31"
        if model is None or "&" not in line or not line.rstrip().endswith(r"\\"):
            continue
        cells = [clean_cell(cell) for cell in line.split("&")]
        if len(cells) < 3:
            continue
        methods = {"BF16", "Dense NVFP4", "W4A16 Marlin", "Sparse BF16",
                   "Sparse NVFP4", "DASO (Max speed)", "DASO (Balanced)"}
        if cells[0] in methods:
            output[model, cells[0]] = cells[1:]
        elif cells[1] in methods:
            output[model, cells[1]] = cells[2:]
    return output


def assert_cell(errors: list[str], rows: dict[tuple[str, str], list[str]], model: str,
                method: str, column: int, expected: str, label: str) -> None:
    actual = rows.get((model, method), [])
    if len(actual) <= column or actual[column] != expected:
        got = actual[column] if len(actual) > column else "<missing>"
        errors.append(f"{label}: expected source-derived {expected}, got {got}")


def parse_ablation_rows(table: str) -> dict[tuple[str, str], str]:
    category = None
    output = {}
    for line in table.splitlines():
        match = re.search(r"\\multirow\{\d+\}\{\*\}\{([^}]+)\}", line)
        if match:
            category = match.group(1)
        if "&" not in line or not line.rstrip().endswith(r"\\"):
            continue
        cells = [cell.strip().replace(r"\\", "") for cell in line.split("&")]
        if len(cells) != 3 or cells[1] == "Variant":
            continue
        if cells[0] and not cells[0].startswith(r"\multirow"):
            category = clean_cell(cells[0])
        if category:
            output[category, clean_cell(cells[1])] = clean_cell(cells[2])
    return output


def numeric_checks(root: Path, errors: list[str]) -> dict[str, int]:
    paper_text = (root / "paper_source/wu.tex").read_text()
    generated = expected_tables(root)
    for path, content in generated.items():
        if not path.is_file() or path.read_text() != content:
            errors.append(f"generated table is out of date: {path.relative_to(root)}")

    main_expected = main_table_values(root)
    main_canonical = parse_rows((root / "tables/main_table_final.tex").read_text())
    main_paper = parse_rows(table_environment(paper_text, "tab:overall_results"))
    for (model, method), values in main_expected.items():
        for column, value in enumerate(values):
            expected = format2(value)
            label = f"main/{model}/{method}/column_{column}"
            assert_cell(errors, main_canonical, model, method, column, expected,
                        f"canonical {label}")
            assert_cell(errors, main_paper, model, method, column, expected,
                        f"paper {label}")

    pro_expected = pro6000_values(root)
    pro_canonical = parse_rows((root / "tables/cross_platform_pro6000_final.tex").read_text(),
                               default_model="pro6000")
    pro_paper = parse_rows(table_environment(paper_text, "tab:pro6000_results"),
                           default_model="pro6000")
    for method, values in pro_expected.items():
        for column, value in enumerate(values):
            expected = format2(value)
            label = f"pro6000/{method}/column_{column}"
            assert_cell(errors, pro_canonical, "pro6000", method, column, expected,
                        f"canonical {label}")
            assert_cell(errors, pro_paper, "pro6000", method, column, expected,
                        f"paper {label}")

    ablation_expected = {(category, variant): value
                         for category, variant, value in ablation_values(root)}
    ablation_canonical = parse_ablation_rows((root / "tables/ablation.tex").read_text())
    ablation_paper = parse_ablation_rows(table_environment(paper_text, "tab:modeling-ablation"))
    for key, expected in ablation_expected.items():
        if ablation_canonical.get(key) != expected:
            errors.append(f"canonical ablation {key}: expected {expected}, got {ablation_canonical.get(key)}")
        if ablation_paper.get(key) != expected:
            errors.append(f"paper ablation {key}: expected {expected}, got {ablation_paper.get(key)}")

    metrics = root / "ablations/data/key_metrics.csv"
    overlap = select(metrics, category="Runtime overlap", variant="Total transfer work")
    exposed = select(metrics, category="Runtime overlap", variant="Exposed copy stall")
    hidden = select(metrics, category="Runtime overlap", variant="Hidden ratio")
    derived = 1.0 - float(exposed["value"]) / float(overlap["value"])
    if abs(derived - float(hidden["value"])) > 1e-12:
        errors.append("runtime overlap Hidden ratio is inconsistent with transfer/exposed time")
    for label, value in (("total transfer", float(overlap["value"])),
                         ("exposed copy", float(exposed["value"]))):
        if f"{value:.2f}" not in paper_text:
            errors.append(f"paper missing source-derived runtime overlap {label}={value:.2f}")
    if f"{100 * derived:.2f}\\%" not in paper_text:
        errors.append("paper overlap percentage does not match structured ablation data")
    return {"main_table_cells": len(main_expected) * 11 * 2,
            "pro6000_table_cells": len(pro_expected) * 6 * 2,
            "ablation_table_cells": len(ablation_expected) * 2,
            "runtime_overlap_values": 4,
            "generated_table_files": len(generated)}


def path_checks(root: Path, errors: list[str]) -> int:
    suffixes = {".csv", ".json", ".jsonl", ".md", ".tex", ".toml", ".txt", ".yaml", ".yml"}
    forbidden = ("/root/wja", "/home/agent", "/root/workspaces")
    checked = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        checked += 1
        text = path.read_text()
        for prefix in forbidden:
            if prefix in text:
                errors.append(f"machine-local path in {path.relative_to(root)}: {prefix}")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/paper")
    args = parser.parse_args()
    root = args.root.resolve()
    errors = validate_manifest(root / "manifest.json")
    required_rows = {
        "main_rtx5090/llama2_7b_chat/prefill_only/data/complete_results.csv": 1,
        "main_rtx5090/llama2_7b_chat/prefill_decode/data/complete_results.csv": 1,
        "main_rtx5090/llama31_8b_instruct/prefill_decode/data/complete_results.csv": 1,
        "main_rtx5090/llama31_8b_instruct/prefill_only/report/all_results.csv": 1,
        "cross_platform_pro6000/data/selected_results.csv": 1,
        "ablations/data/key_metrics.csv": 1,
        "appendix_pareto_audit/prefill_only/report/all_results.csv": 30,
        "appendix_pareto_audit/prefill_decode/report/all_results.csv": 30,
    }
    for relative, minimum in required_rows.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing required result: {relative}")
        elif len(read_rows(path)) < minimum:
            errors.append(f"too few rows: {relative}")
    structured_checks: dict[str, int] = {}
    try:
        structured_checks = numeric_checks(root, errors)
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f"structured paper validation failed: {exc}")
    sanitized_files = path_checks(root, errors)
    output = {"ok": not errors, "errors": errors,
              "checks": {"structural": len(required_rows),
                         "source_derived_numeric": structured_checks,
                         "sanitized_text_files": sanitized_files,
                         "manifest": "all archived files"}}
    print(json.dumps(output, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
