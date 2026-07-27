#!/usr/bin/env python3
"""Validate every raw sample and derived Pro 6000 decode timing value."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1]
MEASURE = BUNDLE / "measurements/decode_components"
EXPECTED_ROWS = 14
EXPECTED_UUID = "305f915b-c789-ebb0-e184-56b64931412f"
CV_LIMIT = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def main() -> None:
    with (BUNDLE / "data/selected_results.csv").open(newline="") as handle:
        selected = [row for row in csv.DictReader(handle) if row["scenario"] == "prefill_decode"]
    with (MEASURE / "summary.csv").open(newline="") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    if len(selected) != EXPECTED_ROWS or len(aggregate_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got selected={len(selected)}, summary={len(aggregate_rows)}")
    aggregate = {(row["model"], row["source_label"]): row for row in aggregate_rows}
    if len(aggregate) != EXPECTED_ROWS:
        raise RuntimeError("duplicate rows in component summary")

    process_ids: set[int] = set()
    stability = []
    summaries = {}
    for selected_row in selected:
        model, label = selected_row["model"], selected_row["source_label"]
        key = (model, label)
        run = MEASURE / "runs" / model / label
        policy = MEASURE / "policies" / model / f"{label}.json"
        policy_sha = sha256(policy)
        audit = json.loads((run / "checkpoint_audit.json").read_text())
        if audit.get("ok") is not True or audit["policy_sha256"] != policy_sha:
            raise RuntimeError(f"checkpoint audit failed for {model}/{label}")

        values: dict[str, list[float]] = {"ttft": [], "main": []}
        uuids = set()
        local_pids = set()
        for phase, output_seq in (("ttft", 1), ("main", 64)):
            expected_names = {"warmup.json", *(f"measured_{index}.json" for index in range(5))}
            raw_dir = run / "raw" / phase
            actual_names = {path.name for path in raw_dir.glob("*.json")}
            if actual_names != expected_names:
                raise RuntimeError(f"raw sample set mismatch for {model}/{label}/{phase}: {actual_names}")
            for name in sorted(expected_names):
                row = json.loads((raw_dir / name).read_text())
                expected = {
                    "phase": phase, "batch": 8, "input_seq": 2048,
                    "output_seq": output_seq, "generated_tokens": 8 * output_seq,
                    "execution": "one_vllm_process_per_sample",
                    "chunked_prefill_enabled": False, "prefix_caching_enabled": False,
                    "max_num_batched_tokens": 16384,
                }
                for field, wanted in expected.items():
                    if row.get(field) != wanted:
                        raise RuntimeError(f"{model}/{label}/{phase}/{name}: {field}={row.get(field)!r}")
                if row["cuda_device_uuid"] != EXPECTED_UUID:
                    raise RuntimeError(f"unexpected GPU UUID for {model}/{label}: {row['cuda_device_uuid']}")
                pid = int(row["process_id"])
                if pid in process_ids or pid in local_pids:
                    raise RuntimeError(f"process was reused: {pid}")
                local_pids.add(pid)
                uuids.add(row["cuda_device_uuid"])
                if name.startswith("measured_"):
                    values[phase].append(float(row["elapsed_ms"]))
        process_ids.update(local_pids)

        summary = json.loads((run / "summary.json").read_text())
        summaries[key] = summary
        ttft = statistics.median(values["ttft"])
        e2e = statistics.median(values["main"])
        tpot = (e2e - ttft) / 63
        checks = {
            "ttft_median_ms": ttft,
            "e2e_median_ms": e2e,
            "tpot_ms": tpot,
        }
        if summary.get("rtx5090_protocol_match") is not True or summary["policy_sha256"] != policy_sha:
            raise RuntimeError(f"summary provenance failed for {model}/{label}")
        if summary["ttft_measured_ms"] != values["ttft"] or summary["e2e_measured_ms"] != values["main"]:
            raise RuntimeError(f"summary samples differ from raw records for {model}/{label}")
        for field, wanted in checks.items():
            if not close(float(summary[field]), wanted):
                raise RuntimeError(f"derived {field} mismatch for {model}/{label}")
            if not close(float(aggregate[key][field]), wanted):
                raise RuntimeError(f"aggregate {field} mismatch for {model}/{label}")

        ttft_cv = statistics.pstdev(values["ttft"]) / statistics.mean(values["ttft"])
        e2e_cv = statistics.pstdev(values["main"]) / statistics.mean(values["main"])
        if max(ttft_cv, e2e_cv) > CV_LIMIT:
            raise RuntimeError(f"unstable measurements for {model}/{label}: TTFT CV={ttft_cv}, E2E CV={e2e_cv}")
        stability.append({
            "model": model, "label": label,
            "ttft_cv": ttft_cv, "e2e_cv": e2e_cv,
            "ttft_min_ms": min(values["ttft"]), "ttft_max_ms": max(values["ttft"]),
            "e2e_min_ms": min(values["main"]), "e2e_max_ms": max(values["main"]),
        })

    for key, row in aggregate.items():
        dense = summaries[(key[0], "uniform_p00")]
        current = summaries[key]
        expected_speedups = {
            "ttft_speedup_vs_bf16": dense["ttft_median_ms"] / current["ttft_median_ms"],
            "tpot_speedup_vs_bf16": dense["tpot_ms"] / current["tpot_ms"],
            "e2e_speedup_vs_bf16": dense["e2e_median_ms"] / current["e2e_median_ms"],
        }
        for field, wanted in expected_speedups.items():
            if not close(float(row[field]), wanted):
                raise RuntimeError(f"speedup mismatch for {key}: {field}")

    temporary_files = list((MEASURE / "temporary").rglob("*"))
    temporary_files = [path for path in temporary_files if path.is_file()]
    if temporary_files:
        raise RuntimeError(f"temporary checkpoint files remain: {temporary_files[:3]}")
    superseded = MEASURE / "superseded_unstable_ttft/llama31_8b_instruct/point_013"
    if len(list((superseded / "ttft").glob("*.json"))) != 6 or not (superseded / "summary.json").exists():
        raise RuntimeError("superseded unstable TTFT evidence is incomplete")

    result = {
        "ok": True,
        "protocol": "RTX-5090-compatible isolated process per sample",
        "policies": EXPECTED_ROWS,
        "current_raw_samples": len(process_ids),
        "unique_process_ids": len(process_ids),
        "gpu_uuid": EXPECTED_UUID,
        "checkpoint_audits_ok": EXPECTED_ROWS,
        "max_ttft_cv": max(row["ttft_cv"] for row in stability),
        "max_e2e_cv": max(row["e2e_cv"] for row in stability),
        "cv_limit": CV_LIMIT,
        "superseded_unstable_ttft_groups_preserved": 1,
        "temporary_checkpoint_files": 0,
        "stability": stability,
    }
    output = BUNDLE / "validation/decode_components.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
