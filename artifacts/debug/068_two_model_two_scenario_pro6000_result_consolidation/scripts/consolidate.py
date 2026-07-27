#!/usr/bin/env python3
"""Build the immutable Pro 6000 paper-result consolidation from 064--067."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
SOURCE = {
    ("llama2_7b_chat", "prefill_only"): ROOT / "artifacts/debug/064_llama2_pro6000_prefill_only",
    ("llama2_7b_chat", "prefill_decode"): ROOT / "artifacts/debug/065_llama2_pro6000_prefill_decode",
    ("llama31_8b_instruct", "prefill_decode"): ROOT / "artifacts/debug/066_llama31_pro6000_prefill_decode",
    ("llama31_8b_instruct", "prefill_only"): ROOT / "artifacts/debug/067_llama31_pro6000_prefill_only",
}
MODEL_DISPLAY = {
    "llama2_7b_chat": r"\shortstack[c]{Llama-2\\7B-Chat}",
    "llama31_8b_instruct": r"\shortstack[c]{Llama-3.1\\8B-Instruct}",
}
METHODS = [
    ("BF16", "uniform_p00"),
    ("Dense NVFP4", "uniform_p01"),
    ("Sparse BF16", "uniform_p02"),
    ("Sparse NVFP4", "uniform_p03"),
    ("W4A16 Marlin", "uniform_p04"),
]
QUALITY_COLUMNS = {
    "prefill_only": ["arc_challenge", "arc_easy", "winogrande", "mmlu"],
    "prefill_decode": ["cnn_dm_1000_rougeL_percent", "dsum_rougeL_percent", "IWSLT_sacre_bleu"],
}
BALANCED_FLOORS = {
    "prefill_only": (0.95, 0.90),
    "prefill_decode": (0.875, 0.75),
}
COMPONENT_SUMMARY = OUT / "measurements/decode_components/summary.csv"
COMPONENT_VALIDATION = OUT / "validation/decode_components.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def label(row: dict[str, str], scenario: str) -> str:
    return row["policy"] if scenario == "prefill_only" else row["label"]


def speed(row: dict[str, str], scenario: str) -> float:
    key = "measured_speedup" if scenario == "prefill_only" else "measured_speedup_vs_dense"
    return float(row[key])


def task_complete(row: dict[str, str], scenario: str) -> bool:
    return all(row.get(column, "") != "" for column in QUALITY_COLUMNS[scenario])


def retentions(row: dict[str, str], dense: dict[str, str], scenario: str) -> list[float]:
    return [float(row[column]) / float(dense[column]) for column in QUALITY_COLUMNS[scenario]]


def select(rows: list[dict[str, str]], scenario: str) -> tuple[dict[str, str], dict[str, str]]:
    dense = next(row for row in rows if label(row, scenario) == "uniform_p00")
    candidates = [row for row in rows if row["family"] == "ours" and task_complete(row, scenario)]
    max_speed = max(candidates, key=lambda row: speed(row, scenario))
    mean_floor, min_floor = BALANCED_FLOORS[scenario]
    qualified = []
    for row in candidates:
        ratios = retentions(row, dense, scenario)
        if sum(ratios) / len(ratios) >= mean_floor and min(ratios) >= min_floor:
            qualified.append(row)
    if not qualified:
        raise RuntimeError(f"no balanced candidate for {scenario}")
    return max_speed, max(qualified, key=lambda row: speed(row, scenario))


def fmt_quality(row: dict[str, str], scenario: str) -> list[str]:
    factor = 100.0 if scenario == "prefill_only" else 1.0
    return [f"{float(row[column]) * factor:.2f}" for column in QUALITY_COLUMNS[scenario]]


def build_latex(table_rows: dict[tuple[str, str], list[dict[str, str]]], components_available: bool) -> str:
    source_tex = ROOT / "artifacts/debug/060_two_model_two_scenario_result_consolidation/result_v2.tex"
    text = source_tex.read_text()
    caption = "Quality and speedup on the RTX~5090 and RTX~PRO~6000. TTFT, TPOT, and E2E speedups are measured relative to BF16 on the same GPU."
    if not components_available:
        caption = "Quality and speedup on the RTX~5090 and RTX~PRO~6000, relative to BF16 on the same GPU. RTX~5090 reports measured TTFT/TPOT/E2E; RTX~PRO~6000 reports measured prefill-only TTFT and decode E2E, while decode TTFT/TPOT decomposition was not retained (marked --)."
    text = text.replace(
        "Quality and speedup on the RTX~5090 and RTX~PRO~6000. TTFT, TPOT, and E2E speedups are measured relative to BF16 on the same GPU.",
        caption,
    )
    start = text.index(r"\multicolumn{13}{l}{\textbf{(b) NVIDIA RTX PRO 6000}}")
    end = text.index(r"\bottomrule", start)
    lines = [r"\multicolumn{13}{l}{\textbf{(b) NVIDIA RTX PRO 6000}} \\", r"\midrule"]
    for model_index, model in enumerate(("llama2_7b_chat", "llama31_8b_instruct")):
        prefill = {row["table_role"]: row for row in table_rows[(model, "prefill_only")]}
        decode = {row["table_role"]: row for row in table_rows[(model, "prefill_decode")]}
        roles = [name for name, _ in METHODS] + ["Ours (Max speed)", "Ours (Balanced)"]
        for index, role in enumerate(roles):
            prefix = rf"\multirow{{7}}{{*}}{{{MODEL_DISPLAY[model]}}}" if index == 0 else ""
            pre = prefill[role]
            dec = decode[role]
            pquality = [pre[column] for column in ("metric_1", "metric_2", "metric_3", "metric_4")]
            dquality = [dec[column] for column in ("metric_1", "metric_2", "metric_3")]
            lines.append(
                (f"{prefix}\n" if prefix else "")
                + f"& {role:<19} & "
                + " & ".join(pquality)
                + f" & {pre['speedup']} & "
                + " & ".join(dquality)
                + f" & {dec['ttft_speedup']} & {dec['tpot_speedup']} & {dec['speedup']} \\\\"
            )
        if model_index == 0:
            lines.append(r"\midrule")
    replacement = "\n".join(lines) + "\n"
    return text[:start] + replacement + text[end:]


def main() -> None:
    data_dir = OUT / "data"
    validation_dir = OUT / "validation"
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    provenance = {}
    selected_records = []
    table_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    components = {}
    if COMPONENT_SUMMARY.exists():
        components = {
            (row["model"], row["source_label"]): row
            for row in load_rows(COMPONENT_SUMMARY)
        }
        if len(components) != 14:
            raise RuntimeError(f"expected 14 decode component rows, got {len(components)}")
        component_validation = json.loads(COMPONENT_VALIDATION.read_text())
        if component_validation.get("ok") is not True:
            raise RuntimeError("decode component validation failed")
    for (model, scenario), bundle in SOURCE.items():
        validation_path = bundle / "validation/all.json"
        validation = json.loads(validation_path.read_text())
        if validation.get("ok") is not True:
            raise RuntimeError(f"source validation failed: {bundle}")
        validation_snapshot = data_dir / f"{model}_{scenario}_validation.json"
        shutil.copyfile(validation_path, validation_snapshot)
        source_csv = bundle / "results/complete_results.csv"
        snapshot = data_dir / f"{model}_{scenario}_complete_results.csv"
        shutil.copyfile(source_csv, snapshot)
        if sha256(source_csv) != sha256(snapshot):
            raise RuntimeError(f"snapshot mismatch: {source_csv}")
        provenance[f"{model}/{scenario}"] = {
            "bundle": str(bundle.relative_to(ROOT)),
            "source_csv": str(source_csv.relative_to(ROOT)),
            "source_csv_sha256": sha256(source_csv),
            "source_validation": str(validation_path.relative_to(ROOT)),
            "source_validation_sha256": sha256(validation_path),
            "source_validation_ok": True,
        }

        rows = load_rows(source_csv)
        by_label = {label(row, scenario): row for row in rows}
        max_speed, balanced = select(rows, scenario)
        chosen = METHODS + [
            ("Ours (Max speed)", label(max_speed, scenario)),
            ("Ours (Balanced)", label(balanced, scenario)),
        ]
        dense = by_label["uniform_p00"]
        current = []
        for role, source_label in chosen:
            row = by_label[source_label]
            qualities = fmt_quality(row, scenario)
            ratios = retentions(row, dense, scenario)
            component = components.get((model, source_label)) if scenario == "prefill_decode" else None
            measured_speedup = float(component["e2e_speedup_vs_bf16"]) if component else speed(row, scenario)
            record = {
                "model": model,
                "scenario": scenario,
                "table_role": role,
                "source_label": source_label,
                "family": row["family"],
                "actual_delta_nll": row["actual_delta_nll"],
                "source_median_ms": row["median_ms"],
                "median_ms": component["e2e_median_ms"] if component else row["median_ms"],
                "metric_1": qualities[0],
                "metric_2": qualities[1],
                "metric_3": qualities[2],
                "metric_4": qualities[3] if scenario == "prefill_only" else "",
                "speedup": f"{measured_speedup:.2f}",
                "ttft_speedup": (f"{float(component['ttft_speedup_vs_bf16']):.2f}" if component else ("1.00" if scenario == "prefill_decode" and role == "BF16" else ("--" if scenario == "prefill_decode" else f"{speed(row, scenario):.2f}"))),
                "tpot_speedup": (f"{float(component['tpot_speedup_vs_bf16']):.2f}" if component else ("1.00" if scenario == "prefill_decode" and role == "BF16" else "--")),
                "quality_mean_retention": f"{sum(ratios) / len(ratios):.6f}",
                "quality_min_retention": f"{min(ratios):.6f}",
            }
            current.append(record)
            selected_records.append(record)
        table_rows[(model, scenario)] = current

    fields = list(selected_records[0])
    with (data_dir / "selected_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_records)

    selection = {
        "metric_schema": {
            "prefill_only": ["ARC-Challenge accuracy (%)", "ARC-Easy accuracy (%)", "WinoGrande accuracy (%)", "MMLU accuracy (%)"],
            "prefill_decode": ["CNN/DailyMail ROUGE-L (%)", "DialogSum ROUGE-L (%)", "IWSLT17 SacreBLEU"],
        },
        "rules": {
            "max_speed": "fastest ours row with every table quality metric measured",
            "balanced_prefill_only": "fastest ours row with mean BF16-relative retention >= 0.95 and minimum retention >= 0.90",
            "balanced_prefill_decode": "fastest ours row with mean BF16-relative retention >= 0.875 and minimum retention >= 0.75",
        },
        "decode_component_timing": ("068 isolated-process O=1/O=64 measurements matching the RTX 5090 protocol" if components else "065/066 retained measured E2E only; non-BF16 TTFT/TPOT are intentionally --"),
        "provenance": provenance,
        "ours": [record for record in selected_records if record["family"] == "ours"],
    }
    (data_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (OUT / "result_v3.tex").write_text(build_latex(table_rows, bool(components)))

    tex = (OUT / "result_v3.tex").read_text()
    if tex.count("NVIDIA RTX PRO 6000") != 1:
        raise RuntimeError("unexpected Pro 6000 section count")
    if tex.count("Ours (Max speed)") != 4 or tex.count("Ours (Balanced)") != 4:
        raise RuntimeError("unexpected ours row count")
    result = {
        "ok": True,
        "source_bundles": 4,
        "source_validations_ok": 4,
        "selected_table_rows": len(selected_records),
        "ours_selected_rows": 8,
        "source_snapshots_sha256_match": True,
        "result_v3_pro6000_quality_and_primary_speed_complete": True,
        "decode_ttft_tpot_status": ("complete: 14 RTX-5090-compatible isolated-process O=1/O=64 measurements" if components else "not retained by 065/066; explicitly marked -- for non-BF16 rows"),
    }
    if components:
        result["decode_component_policies"] = component_validation["policies"]
        result["decode_component_raw_samples"] = component_validation["current_raw_samples"]
        result["decode_component_unique_process_ids"] = component_validation["unique_process_ids"]
        result["decode_component_gpu_uuid"] = component_validation["gpu_uuid"]
        result["decode_component_max_ttft_cv"] = component_validation["max_ttft_cv"]
        result["decode_component_max_e2e_cv"] = component_validation["max_e2e_cv"]
    (validation_dir / "all.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
