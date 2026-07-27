#!/usr/bin/env python3
"""Machine-readable stage validation for the isolated 067 workflow."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import BUNDLE, RESULTS, RUN, VALIDATION, sha256, write_json


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate(stage: str) -> dict:
    errors = []
    details = {}
    if stage in {"bootstrap", "all"}:
        manifest = json.loads((RUN / "policies/prefill_only/manifest.json").read_text())
        sample = RUN / "samples/wikitext_2048_targets.pt"
        provenance = json.loads((RUN / "bootstrap_provenance.json").read_text())
        details["policies"] = len(manifest); details["sample_sha256"] = sha256(sample)
        if (len(manifest) != 72 or provenance.get("sample_shape") != [100, 2049]
                or provenance.get("sample_seed") != 86
                or details["sample_sha256"] != "c201cbe60ff361f51956f1385ebb9b2e09250d02fc53b1c88c0e80f7e8293e18"
                or details["sample_sha256"] != provenance.get("sample_sha256")):
            errors.append("bootstrap inputs differ from the frozen design")
    if stage in {"isolation", "all"}:
        forbidden = ("artifacts/debug/038_", "artifacts/debug/046_", "artifacts/debug/054_",
                     "artifacts/debug/058_", "artifacts/debug/063_", "artifacts/debug/064_")
        hits = []
        for path in (BUNDLE / "scripts").glob("*"):
            if path.name in {Path(__file__).name, "copy_canonical.py"}:
                continue
            if path.suffix not in {".py", ".sh"}:
                continue
            text = path.read_text()
            hits.extend(f"{path.name}:{needle}" for needle in forbidden if needle in text)
        details["forbidden_runtime_references"] = hits
        if hits:
            errors.append("scripts retain historical debug runtime references")
    if stage in {"canonical", "all"}:
        metadata = {method: json.loads((RUN / f"canonical/prepared/{method}/metadata.json").read_text())
                    for method in ("sparse_bf16", "sparse_nvfp4")}
        verification = json.loads((RUN / "canonical/verification.json").read_text())
        if metadata["sparse_nvfp4"].get("sparse_nvfp4_prequant_only") is not True:
            errors.append("sparse_nvfp4 is not prequant-only")
        if metadata["sparse_bf16"].get("sparse_nvfp4_prequant_only") is not False:
            errors.append("sparse_bf16 metadata is inconsistent")
        for method in metadata:
            if (metadata[method].get("selected_modules") != 224
                    or metadata[method].get("compressed_modules") != 224
                    or metadata[method].get("skipped")
                    or verification.get(method, {}).get("checked_linear_weights") != 224):
                errors.append(f"{method} canonical coverage is incomplete")
        details["canonical"] = {method: sha256(RUN / f"canonical/prepared/{method}/model.pt") for method in metadata}
    if stage in {"local-errors", "all"}:
        provenance = json.loads((RUN / "local_errors/provenance.json").read_text())
        expected_counts = {"dense_nvfp4": 224, "sparse_bf16": 224, "sparse_nvfp4": 224}
        details["local_error_rows_by_method"] = provenance.get("rows_by_method")
        if (provenance.get("rows_by_method") != expected_counts
                or provenance.get("output_sha256") != sha256(RUN / "local_errors/module_method_errors.csv")):
            errors.append("local-error table coverage or provenance is invalid")
    if stage in {"nll", "all"}:
        rows = csv_rows(RUN / "calibration/nll/prefill_only.csv")
        audits = list((RUN / "calibration/audits").glob("p[0-9][0-9].json"))
        details["nll_rows"] = len(rows); details["checkpoint_audits"] = len(audits)
        if (len(rows) != 72 or len(audits) != 72
                or {row["policy_id"] for row in rows} != {f"p{i:02d}" for i in range(72)}):
            errors.append("NLL table is not complete")
    if stage in {"profile", "all"}:
        frozen = json.loads((RUN / "pareto/frozen_quality_model.json").read_text())
        metadata = json.loads((RUN / "kernel_profile/metadata.json").read_text())
        expected_shapes = [[16384, 6144, 4096], [16384, 4096, 4096],
                           [16384, 28672, 4096], [16384, 4096, 14336]]
        profile_rows = csv_rows(RUN / "kernel_profile/exact/targeted_profile.csv")
        expected = str((RUN / "kernel_profile/modeling").resolve())
        details["predictor_root"] = frozen.get("predictor_root")
        details["profile_shapes"] = metadata.get("shapes")
        details["profile_action_rows"] = len(profile_rows)
        if frozen.get("predictor_root") != expected:
            errors.append("solver did not freeze the 067 predictor")
        if frozen.get("one_time_weight_conversion_excluded") is not True:
            errors.append("solver speed objective includes one-time weight conversion")
        if metadata.get("shapes") != expected_shapes or len(profile_rows) != 20:
            errors.append("kernel profile does not contain the four exact Llama3 prefill shapes")
    if stage in {"closure", "all"}:
        labels = [path.name for path in (RUN / "closure").iterdir() if path.is_dir()]
        points = csv_rows(RUN / "pareto/pareto_points.csv")
        expected_labels = {f"uniform_p{i:02d}" for i in range(5)} | {
            f"point_{int(row['point_index']):03d}" for row in points}
        incomplete = []
        gpu_ids = set()
        for label in sorted(expected_labels):
            base = RUN / "closure" / label
            measured = sorted((base / "speed/raw").glob("measured_[0-9].json"))
            raw_paths = [base / "speed/raw/warmup.json", *measured]
            if (not (base / "nll.json").is_file() or not (base / "checkpoint_audit.json").is_file()
                    or len(measured) != 5 or any(not path.is_file() for path in raw_paths)):
                incomplete.append(label)
                continue
            nll = json.loads((base / "nll.json").read_text())
            audit = json.loads((base / "checkpoint_audit.json").read_text())
            raws = [json.loads(path.read_text()) for path in raw_paths]
            if (len(nll.get("blocks", [])) != 100 or nll.get("runtime", {}).get("chunked_prefill_enabled") is not False
                    or audit.get("prune") is not False
                    or any((row.get("batch"), row.get("input_seq"), row.get("output_seq"),
                            row.get("chunked_prefill_enabled"), row.get("prefix_caching_enabled"),
                            row.get("max_num_batched_tokens"), row.get("single_process_repeats"))
                           != (8, 2048, 1, False, False, 16392, True) for row in raws)
                    or len({row.get("benchmark_process_id") for row in raws}) != 1):
                incomplete.append(label)
            gpu_ids.update(row.get("cuda_device_uuid") for row in raws)
        details["closure_labels"] = labels; details["closure_incomplete"] = incomplete
        details["closure_gpu_uuids"] = sorted(str(value) for value in gpu_ids)
        if incomplete or set(labels) != expected_labels or len(gpu_ids) != 1:
            errors.append("closure is incomplete")
    if stage in {"tasks", "all"}:
        data_manifest = json.loads((BUNDLE / "cache/task_data_manifest.json").read_text())
        selection = json.loads((RUN / "tasks/selection.json").read_text())["selected"]
        incomplete = []
        for label in selection:
            for task in ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu"):
                result_path = RUN / "pareto/validation/tasks" / label / task / "full/result.json"
                if not result_path.is_file():
                    incomplete.append(f"{label}:{task}")
                    continue
                result = json.loads(result_path.read_text())
                guard = result.get("phase_guard_counts", {})
                if (result.get("limit") is not None
                        or guard.get("model_generate_calls", 0) < 1
                        or guard.get("prefill_resets") != guard.get("model_generate_calls") - 1):
                    incomplete.append(f"{label}:{task}:phase_guard")
        details["task_incomplete"] = incomplete
        missing_audits = [label for label in selection if not (RUN / "pareto/validation/tasks" / label / "checkpoint_audit.json").is_file()]
        details["task_missing_checkpoint_audits"] = missing_audits
        details["task_data_complete"] = data_manifest.get("complete")
        if incomplete or missing_audits or data_manifest.get("complete") is not True:
            errors.append("full task evaluation is incomplete")
    if stage in {"results", "all"}:
        path = RESULTS / "complete_results.csv"
        figures = list((RESULTS / "figures").glob("pareto_speed_vs_*.png"))
        details["result_figures"] = len(figures)
        if not path.is_file() or not csv_rows(path) or len(figures) != 6:
            errors.append("final result table missing")
    return {"stage": stage, "ok": not errors, "errors": errors, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bootstrap", "isolation", "canonical", "local-errors", "nll", "profile", "closure", "tasks", "results", "all"))
    args = parser.parse_args()
    try:
        report = validate(args.stage)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        report = {"stage": args.stage, "ok": False, "errors": [f"{type(exc).__name__}: {exc}"], "details": {}}
    write_json(VALIDATION / f"{args.stage}.json", report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
