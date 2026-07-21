#!/usr/bin/env python3
"""Audit bootstrap, canonical, full retained, or smoke artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import BUNDLE, SRC054, SRC056, SRC060, env_path, sha256, write_json


def csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def record_file(checks: list[dict[str, object]], name: str, path: Path, *, rows: int | None = None) -> None:
    item: dict[str, object] = {"name": name, "path": str(path), "exists": path.is_file()}
    if path.is_file():
        item["bytes"] = path.stat().st_size
        if rows is not None:
            item["rows"] = csv_rows(path)
            item["expected_rows"] = rows
            item["ok"] = item["rows"] == rows
        else:
            item["ok"] = True
    else:
        item["ok"] = False
    checks.append(item)


def bootstrap_checks(root: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    provenance = root / "bootstrap_provenance.json"
    record_file(checks, "bootstrap provenance", provenance)
    if provenance.is_file():
        payload = json.loads(provenance.read_text())
        for row in payload.get("files", []):
            target = Path(row["target"])
            expected = row.get("target_sha256", row.get("sha256"))
            checks.append({"name": target.name, "path": str(target), "ok": target.is_file() and sha256(target) == expected})
    for scenario in ("prefill_only", "prefill_decode"):
        policy_dir = root / scenario / "policies" / scenario
        count = len(list(policy_dir.glob("p[0-9][0-9].json")))
        checks.append({"name": f"{scenario} policy count", "path": str(policy_dir), "count": count, "expected": 72, "ok": count == 72})
    return checks


def canonical_checks(canonical: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for method in ("sparse_bf16", "sparse_nvfp4"):
        state = canonical / method / "model.pt"
        checks.append({"name": f"canonical {method}", "path": str(state), "exists": state.is_file(), "bytes": state.stat().st_size if state.is_file() else 0, "ok": state.is_file() and state.stat().st_size > 10_000_000_000})
    return checks


def retained_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    record_file(checks, "054 prefill-only NLL", SRC054 / "llama2_7b_chat/nll/prefill_only.csv", rows=72)
    record_file(checks, "054 local errors", SRC054 / "llama2_7b_chat/local_errors/module_method_errors.csv")
    record_file(checks, "054 fitted quality metrics", SRC054 / "llama2_7b_chat/reports/quality/metrics.json")
    record_file(checks, "054 measured paper methods", SRC054 / "llama2_7b_chat/pareto/paper/all_methods_measured.csv")
    record_file(checks, "056 prefill-decode NLL", SRC056 / "llama2_7b_chat/nll/prefill_decode.csv", rows=72)
    record_file(checks, "056 phase quality metrics", SRC056 / "llama2_7b_chat/reports/quality_coverage_holdout/metrics.json")
    record_file(checks, "056 closure", SRC056 / "llama2_7b_chat/pareto/closure_summary.csv", rows=10)
    record_file(checks, "056 task-quality long table", SRC056 / "llama2_7b_chat/task_quality/summary.csv", rows=24)
    record_file(checks, "060 prefill-only complete", SRC060 / "llama2_7b_chat/prefill_only/data/complete_results.csv")
    record_file(checks, "060 prefill-decode complete", SRC060 / "llama2_7b_chat/prefill_decode/data/complete_results.csv")
    return checks


def smoke_checks(root: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for policy in ("p00", "p01", "p02", "p71"):
        base = root / "validation/smoke" / policy
        record_file(checks, f"{policy} checkpoint manifest", base / "checkpoint/phase_hetero_manifest.json")
        record_file(checks, f"{policy} prefill NLL", base / "prefill_only_nll.json")
        record_file(checks, f"{policy} decode NLL", base / "prefill_decode_nll.json")
        record_file(checks, f"{policy} speed", base / "speed.json")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bootstrap", "canonical", "retained", "smoke"))
    parser.add_argument("--run-root", type=Path, default=env_path("COSPAQ_RUN_ROOT"))
    parser.add_argument("--canonical-dir", type=Path, default=env_path("COSPAQ_CANONICAL_DIR"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "bootstrap":
        checks = bootstrap_checks(args.run_root)
    elif args.stage == "canonical":
        checks = canonical_checks(args.canonical_dir)
    elif args.stage == "retained":
        checks = retained_checks()
    else:
        checks = smoke_checks(args.run_root)
    report = {"stage": args.stage, "ok": all(row.get("ok", False) for row in checks), "checks": checks}
    output = args.output or BUNDLE / f"validation/{args.stage}_audit.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
