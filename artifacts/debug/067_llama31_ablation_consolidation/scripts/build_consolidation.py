#!/usr/bin/env python3
"""Collect authoritative ablation artifacts and write a provenance manifest."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
SRC64 = ROOT / "artifacts/debug/064_llama31_modeling_ablation"
SRC65 = ROOT / "artifacts/debug/065_llama31_quality_calibration_size_sensitivity"
SRC66 = ROOT / "artifacts/debug/066_llama31_vllm_overlap_ablation"
SRC57 = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto"


FILES: list[tuple[Path, str]] = [
    # Modeling data.
    (SRC64 / "quality/metrics.csv", "data/modeling/quality_metrics.csv"),
    (SRC64 / "quality/predictions.csv", "data/modeling/quality_predictions.csv"),
    (SRC64 / "quality/validation.json", "data/modeling/quality_validation.json"),
    (SRC64 / "quality/model_local_sum.json", "data/modeling/model_local_sum.json"),
    (SRC64 / "quality/model_structure_aware.json", "data/modeling/model_structure_aware.json"),
    (SRC64 / "quality/model_full_phase_aware.json", "data/modeling/model_full_phase_aware.json"),
    (SRC64 / "speed/kernel_generalization_metrics.csv", "data/modeling/kernel_generalization_metrics.csv"),
    (SRC64 / "speed/kernel_generalization_predictions.csv", "data/modeling/kernel_generalization_predictions.csv"),
    (SRC64 / "speed/kernel_generalization_validation.json", "data/modeling/kernel_generalization_validation.json"),
    (SRC64 / "speed/local_metrics.csv", "data/modeling/local_latency_metrics.csv"),
    (SRC64 / "speed/local_predictions.csv", "data/modeling/local_latency_predictions.csv"),
    (SRC64 / "speed/validation.json", "data/modeling/latency_validation.json"),
    (SRC64 / "speed/metrics.csv", "data/modeling/e2e_latency_metrics.csv"),
    (SRC64 / "speed/predictions.csv", "data/modeling/e2e_latency_predictions.csv"),
    # Modeling figures, tables, and documents.
    (SRC64 / "report/modeling_ablation.png", "figures/modeling_ablation.png"),
    (SRC64 / "report/modeling_ablation.pdf", "figures/modeling_ablation.pdf"),
    (SRC64 / "report/modeling_ablation_detailed.png", "figures/modeling_ablation_detailed.png"),
    (SRC64 / "report/modeling_ablation_detailed.pdf", "figures/modeling_ablation_detailed.pdf"),
    (SRC64 / "report/ablation_tables.tex", "tables/modeling_ablation.tex"),
    (SRC64 / "report/modeling_design_and_ablation.md", "docs/modeling_design_and_ablation.md"),
    (SRC64 / "report/paper_text.md", "docs/modeling_paper_text.md"),
    (SRC64 / "report/summary.md", "docs/modeling_summary.md"),
    (SRC64 / "report/detailed_caption.md", "docs/modeling_detailed_caption.md"),
    (SRC64 / "report/reproducibility.json", "docs/modeling_reproducibility.json"),
    # Calibration-size data and figures.
    (SRC65 / "results/learning_curve_repeats.csv", "data/calibration/learning_curve_repeats.csv"),
    (SRC65 / "results/learning_curve_summary.csv", "data/calibration/learning_curve_summary.csv"),
    (SRC65 / "results/validation.json", "data/calibration/validation.json"),
    (SRC65 / "extension/nll/extension.csv", "data/calibration/extension_nll.csv"),
    (SRC65 / "extension/policy_audit.json", "data/calibration/extension_policy_audit.json"),
    (SRC65 / "extension/policies/manifest.json", "data/calibration/extension_policy_manifest.json"),
    (SRC65 / "report/calibration_size_sensitivity.png", "figures/calibration_size_sensitivity.png"),
    (SRC65 / "report/calibration_size_sensitivity.pdf", "figures/calibration_size_sensitivity.pdf"),
    (SRC65 / "report/calibration_size_sensitivity_full.png", "figures/calibration_size_sensitivity_full.png"),
    (SRC65 / "report/calibration_size_sensitivity_full.pdf", "figures/calibration_size_sensitivity_full.pdf"),
    (SRC65 / "report/paper_text.md", "docs/calibration_paper_text.md"),
    (SRC65 / "report/summary.md", "docs/calibration_summary.md"),
    # Runtime data, figures, tables, and documents.
    (SRC66 / "raw_samples.csv", "data/runtime/raw_samples.csv"),
    (SRC66 / "run_metadata.json", "data/runtime/run_metadata.json"),
    (SRC66 / "config_audit.json", "data/runtime/config_audit.json"),
    (SRC66 / "report/summary.json", "data/runtime/e2e_summary.json"),
    (SRC66 / "report/overlap_profile.json", "data/runtime/overlap_profile.json"),
    (SRC66 / "report/overlap_ablation.png", "figures/overlap_ablation.png"),
    (SRC66 / "report/overlap_ablation.pdf", "figures/overlap_ablation.pdf"),
    (SRC66 / "report/ablation_table.tex", "tables/runtime_e2e_ablation.tex"),
    (SRC66 / "report/overlap_profile_table.tex", "tables/runtime_overlap_profile.tex"),
    (SRC66 / "report/summary.md", "docs/runtime_summary.md"),
    (SRC66 / "report/overlap_profile.md", "docs/runtime_overlap_profile.md"),
    (SRC66 / "report/paper_text.md", "docs/runtime_paper_text.md"),
    (SRC66 / "config_audit.json", "docs/runtime_config_audit.json"),
    (SRC66 / "checkpoint/point_011/phase_hetero_export_provenance.json",
     "data/runtime/policy/phase_hetero_export_provenance.json"),
    (SRC57 / "llama31_8b_instruct/pareto/policies/point_011.json",
     "data/runtime/policy/point_011.json"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_tree(source: Path, destination: str, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in sorted(source.glob(pattern)):
            if path.is_file():
                FILES.append((path, f"{destination}/{path.relative_to(source)}"))


def copy_files() -> list[dict[str, object]]:
    add_tree(SRC65 / "extension/policies", "data/calibration/extension_policies", ("e*.json",))
    add_tree(SRC65 / "extension/logs", "data/calibration/extension_logs", ("e*.log",))
    add_tree(SRC66 / "profile", "data/runtime/profile", ("sample_*.json",))
    add_tree(SRC66 / "raw", "data/runtime/raw", ("**/*.json", "**/*.log"))
    add_tree(SRC64 / "scripts", "reproduction/modeling", ("*.py",))
    add_tree(SRC65 / "scripts", "reproduction/calibration", ("*.py",))
    add_tree(SRC66 / "scripts", "reproduction/runtime", ("*.py", "*.sh"))

    records = []
    seen: set[str] = set()
    for source, relative in FILES:
        if relative in seen:
            continue
        seen.add(relative)
        if not source.exists():
            raise FileNotFoundError(source)
        target = EXP / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"hash mismatch after copying {source}")
        records.append({
            "source": str(source.relative_to(ROOT)),
            "destination": relative,
            "bytes": source.stat().st_size,
            "sha256": source_hash,
        })
    return records


def write_key_metrics() -> None:
    rows: list[dict[str, object]] = []
    kernel = list(csv.DictReader((EXP / "data/modeling/kernel_generalization_metrics.csv").open()))
    names = {"direct_fit": "Direct fit", "roofline": "+ Roofline prior",
             "roofline_shape_corrected": "+ Shape correction"}
    for row in kernel:
        if row["kernel"] == "all":
            rows.append({"category": "Kernel prediction", "variant": names[row["variant"]],
                         "metric": "MAPE", "value": float(row["mape"]), "unit": "fraction"})
    e2e = list(csv.DictReader((EXP / "data/modeling/e2e_latency_metrics.csv").open()))
    for variant, name in (("shape_corrected", "Raw E2E aggregation"),
                          ("policy_calibrated_pava", "+ Monotonic E2E calibration")):
        row = next(item for item in e2e if item["variant"] == variant and item["split"] == "holdout")
        rows.append({"category": "E2E composition", "variant": name,
                     "metric": "MAPE", "value": float(row["mape"]), "unit": "fraction"})
    quality = list(csv.DictReader((EXP / "data/modeling/quality_metrics.csv").open()))
    qnames = {"local_sum": "Local aggregation", "structure_aware": "+ Structure awareness",
              "full_phase_aware": "+ Phase awareness"}
    for row in quality:
        if row["split"] == "holdout":
            rows.append({"category": "Quality prediction", "variant": qnames[row["variant"]],
                         "metric": "Delta-NLL MAE", "value": float(row["mae"]), "unit": "NLL"})
    runtime = json.loads((EXP / "data/runtime/e2e_summary.json").read_text())
    for variant, key in (("Synchronous phase-boundary copy", "synchronous_copy_median_ms"),
                         ("Asynchronous overlapped prefetch", "asynchronous_overlap_median_ms")):
        rows.append({"category": "Runtime E2E", "variant": variant,
                     "metric": "Median latency", "value": runtime[key], "unit": "ms"})
    profile = json.loads((EXP / "data/runtime/overlap_profile.json").read_text())
    for variant, key, unit in (
        ("Total transfer work", "median_total_transfer_work_ms", "ms"),
        ("Exposed copy stall", "median_exposed_copy_stall_ms", "ms"),
        ("Hidden ratio", "median_hidden_ratio", "fraction"),
    ):
        rows.append({"category": "Runtime overlap", "variant": variant,
                     "metric": variant, "value": profile[key], "unit": unit})
    path = EXP / "data/key_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records = copy_files()
    write_key_metrics()
    manifest = {
        "artifact": "067_llama31_ablation_consolidation",
        "model": "Llama3.1-8B-Instruct",
        "source_experiments": [
            "064_llama31_modeling_ablation",
            "065_llama31_quality_calibration_size_sensitivity",
            "066_llama31_vllm_overlap_ablation",
        ],
        "files": records,
    }
    (EXP / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"copied_files": len(records), "bytes": sum(r["bytes"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()

