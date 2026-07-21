#!/usr/bin/env python3
"""Refit the unchanged v2 formula after adding coverage-only training labels."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/048_llama2_prefill_quality_coverage"
PREVIOUS = ROOT / "artifacts/debug/047_llama2_prefill_mechanism_quality_debug"
SOURCE = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat"
sys.path.insert(0, str(PREVIOUS / "scripts"))
from fit_mechanism_proxy import errors, metric, signals  # noqa: E402


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def entries_from(manifest: list[dict], labels: dict[str, dict], group: str) -> list[dict]:
    return [{"id": item["policy_id"], "group": group if item["split"] == "train" else f"{group}_holdout", "policy": json.loads(Path(item["path"]).read_text()), "y": float(labels[item["policy_id"]]["delta_nll"] if "delta_nll" in labels[item["policy_id"]] else labels[item["policy_id"]]["target_delta_nll"])} for item in manifest]


def main() -> None:
    old_manifest = json.loads((SOURCE / "policies/prefill_only/manifest.json").read_text()); old_labels = {row["policy_id"]: row for row in read(SOURCE / "nll/prefill_only.csv")}
    mechanism_manifest = json.loads((PREVIOUS / "manifest.json").read_text()); mechanism_labels = {row["policy_id"]: row for row in read(PREVIOUS / "nll.csv")}
    coverage_manifest = json.loads((DEBUG / "manifest.json").read_text()); coverage_labels = {row["policy_id"]: row for row in read(DEBUG / "nll.csv")}
    entries = []
    entries += [{"id": item["policy_id"], "group": "old_train" if item["split"] == "train" else "old_holdout", "policy": json.loads(Path(item["path"]).read_text()), "y": float(old_labels[item["policy_id"]]["target_delta_nll"])} for item in old_manifest]
    entries += entries_from(mechanism_manifest, mechanism_labels, "mechanism_train")
    entries += entries_from(coverage_manifest, coverage_labels, "coverage_train")
    q_error, s_error = errors("dense_nvfp4"), errors("sparse_bf16")
    q = torch.stack([signals(entry["policy"], q_error, s_error)[0] for entry in entries]); s = torch.stack([signals(entry["policy"], q_error, s_error)[1] for entry in entries]); y = torch.tensor([entry["y"] for entry in entries], dtype=torch.float64)
    train = torch.tensor([not entry["group"].endswith("holdout") for entry in entries]); scale = (q[train].sum((1, 2)) + s[train].sum((1, 2))).mean().clamp(min=1e-12); q, s = q / scale, s / scale
    params = [torch.full((4, 4), .01, dtype=torch.float64, requires_grad=True), torch.full((4, 4), .01, dtype=torch.float64, requires_grad=True), torch.full((4,), .01, dtype=torch.float64, requires_grad=True), torch.full((4,), .01, dtype=torch.float64, requires_grad=True)]
    optimizer = torch.optim.Adam(params, lr=.025)
    for _ in range(5000):
        optimizer.zero_grad(); qw, sw, a, c = [torch.relu(value) for value in params]; qg, sg = q.sum(2), s.sum(2)
        prediction = (q * qw).sum((1, 2)) + (s * sw).sum((1, 2)) + (sg.square() * a).sum(1) + (sg * qg * c).sum(1)
        (((prediction[train] - y[train]).square()).mean() + .0001 * sum(value.square().mean() for value in params)).backward(); optimizer.step()
    with torch.no_grad():
        qw, sw, a, c = [torch.relu(value) for value in params]; qg, sg = q.sum(2), s.sum(2); prediction = (q * qw).sum((1, 2)) + (s * sw).sum((1, 2)) + (sg.square() * a).sum(1) + (sg * qg * c).sum(1)
    rows = [{"policy_id": entry["id"], "group": entry["group"], "actual_delta_nll": entry["y"], "predicted_delta_nll": float(prediction[index]), "residual": entry["y"] - float(prediction[index])} for index, entry in enumerate(entries)]
    report = DEBUG / "report"; report.mkdir(parents=True, exist_ok=True)
    with (report / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {group: metric([row["actual_delta_nll"] for row in rows if row["group"] == group], [row["predicted_delta_nll"] for row in rows if row["group"] == group]) for group in sorted({row["group"] for row in rows})}
    (report / "metrics.json").write_text(json.dumps({"formula": "unchanged Q/S/S2/SQ ReLU proxy", "train_groups": ["old_train", "mechanism_train", "coverage_train"], "metrics": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
