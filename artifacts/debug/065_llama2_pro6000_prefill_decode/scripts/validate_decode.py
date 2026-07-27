#!/usr/bin/env python3
"""Machine-readable stage validation for the isolated 065 workflow."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from common import BUNDLE, PROTOCOL, RESULTS, RUN, VALIDATION, sha256, write_json


def csv_rows(path: Path):
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def validate(stage: str) -> dict:
    errors, details = [], {}
    if stage in {"bootstrap", "all"}:
        manifest = json.loads((RUN / "policies/prefill_decode/manifest.json").read_text())
        sample = RUN / "samples/wikitext_2048_64.pt"
        shape = tuple(torch.load(sample, map_location="cpu", weights_only=True).shape)
        details["bootstrap"] = {"policies": len(manifest), "sample_shape": shape, "sample_sha256": sha256(sample)}
        if len(manifest) != 72 or shape != (300, 2112): errors.append("bootstrap coverage mismatch")
        if any(not str(Path(row["path"]).resolve()).startswith(str(BUNDLE.resolve())) for row in manifest):
            errors.append("manifest points outside 065")
    if stage in {"isolation", "all"}:
        active = ("common.py", "bootstrap.py", "copy_canonical.py", "export_phase_hetero_model.py",
                  "evaluate_decode_nll.py", "run_calibration_nll.py", "fit_phase_quality.py",
                  "profile_kernels.py", "audit_speed_actions.py", "solve_phase_pareto.py",
                  "closure_policy.py", "run_pmpd_tasks.py", "consolidate_decode.py")
        forbidden = []
        for name in active:
            text = (BUNDLE / "scripts" / name).read_text()
            for old in ("artifacts/debug/054_", "artifacts/debug/055_", "artifacts/debug/056_", "artifacts/debug/063_"):
                if old in text: forbidden.append(f"{name}:{old}")
        details["isolation"] = {"active_scripts": len(active), "forbidden_references": forbidden,
                                "allowed_canonical_copy_source": "064"}
        errors.extend(forbidden)
    if stage in {"canonical", "all"}:
        provenance = json.loads((RUN / "canonical/copy_provenance.json").read_text())
        verification = json.loads((RUN / "canonical/verification.json").read_text())
        hashes = {method: sha256(RUN / f"canonical/prepared/{method}/model.pt")
                  for method in ("sparse_bf16", "sparse_nvfp4")}
        details["canonical"] = {"hashes": hashes, "verification": verification}
        for method, digest in hashes.items():
            if provenance["states"][method]["sha256"] != digest: errors.append(f"{method} copy hash mismatch")
    if stage in {"local-errors", "all"}:
        counts = {}
        for phase in ("prefill", "decode"):
            for method in ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours"):
                rows = csv_rows(RUN / f"local_errors/{phase}_{method}.csv")
                counts[f"{phase}/{method}"] = len(rows)
                if len(rows) != 16: errors.append(f"incomplete local errors: {phase}/{method}")
        details["local_errors"] = counts
    if stage in {"nll", "all"}:
        rows = csv_rows(RUN / "calibration/nll/prefill_decode.csv")
        details["nll"] = {"rows": len(rows)}
        if len(rows) != 72: errors.append("NLL must contain 72 rows")
        for row in rows:
            payload = json.loads((RUN / f"calibration/raw/{row['policy_id']}.json").read_text())
            runtime = payload["runtime"]
            if runtime.get("max_num_batched_tokens") != PROTOCOL["teacher_forcing_capacity"]:
                errors.append(f"NLL capacity mismatch: {row['policy_id']}")
            if not runtime.get("phase_trace_events", {}).get("apply_decode"):
                errors.append(f"missing decode trace: {row['policy_id']}")
    if stage in {"profile", "all"}:
        metadata = json.loads((RUN / "kernel_profile/metadata.json").read_text())
        actions = csv_rows(RUN / "speed/action_support.csv")
        bad_decode = [row for row in actions if row["phase"] == "decode" and row["kernel"] == "sparse_nvfp4" and row["supported"] == "True"]
        details["profile"] = {"shapes": len(metadata["shapes"]), "actions": len(actions),
                              "decode_sparse_nvfp4_supported": len(bad_decode)}
        if len(metadata["shapes"]) != 8 or len(actions) != 1280 or bad_decode:
            errors.append("profile/action-support mismatch")
    if stage in {"closure", "all"}:
        points = csv_rows(RUN / "pareto/predicted_points.csv")
        labels = [f"uniform_p{i:02d}" for i in range(5)] + [row["policy_id"] for row in points]
        labels = list(dict.fromkeys(labels)); missing = []
        for label in labels:
            root = RUN / "closure" / label
            try:
                nll = json.loads((root / "nll.json").read_text())
                speed = json.loads((root / "speed/summary.json").read_text())
                raw_paths = [root / "speed/raw/warmup.json"] + [
                    root / f"speed/raw/measured_{index}.json" for index in range(5)]
                raw = [json.loads(path.read_text()) for path in raw_paths]
                process_ids = {row["benchmark_process_id"] for row in raw}
                if (len(nll["blocks"]) != 100 or len(speed["measured_elapsed_ms"]) != 5
                        or not speed["single_process_repeats"] or speed["warmup_iters"] != 1
                        or speed["measured_runs"] != 5
                        or process_ids != {speed["benchmark_process_id"]}
                        or any(not row["single_process_repeats"] for row in raw)):
                    missing.append(label)
            except (FileNotFoundError, KeyError, json.JSONDecodeError): missing.append(label)
        details["closure"] = {"expected": len(labels), "missing": missing}
        if missing: errors.append(f"incomplete closure: {missing}")
    if stage in {"tasks", "all"}:
        summary = json.loads((RUN / "tasks/summary.json").read_text())
        expected = len(summary["selected"]) * 3
        details["tasks"] = {"rows": len(summary["rows"]), "expected": expected}
        if len(summary["rows"]) != expected or any(row["num_samples"] != PROTOCOL_COUNT(row["dataset"]) for row in summary["rows"]):
            errors.append("task coverage mismatch")
    if stage in {"results", "all"}:
        rows = csv_rows(RESULTS / "complete_results.csv")
        figures = list((RESULTS / "figures").glob("*.png"))
        details["results"] = {"rows": len(rows), "figures": len(figures)}
        if not rows or len(figures) < 7: errors.append("final results incomplete")
    return {"stage": stage, "ok": not errors, "errors": errors, "details": details}


def PROTOCOL_COUNT(dataset: str) -> int:
    return {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}[dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bootstrap", "isolation", "canonical", "local-errors", "nll", "profile", "closure", "tasks", "results", "all"))
    args = parser.parse_args(); report = validate(args.stage)
    write_json(VALIDATION / f"{args.stage}.json", report); print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
