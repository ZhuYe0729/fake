#!/usr/bin/env python3
"""Fit the full quality model on nested calibration-policy subsets."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
ABLATION64 = ROOT / "artifacts/debug/064_llama31_modeling_ablation"
sys.path.insert(0, str(ABLATION64 / "scripts"))
import run_quality_ablation as quality  # noqa: E402


SIZES = (5, 8, 12, 18, 27, 36, 45, 54, 60, 64, 66, 72)
ANCHORS = ("p00", "p01", "p02", "p03", "p04")
RANDOM_SEEDS = tuple(range(1001, 1011))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def coverage_order(
    policy_ids: list[str], anchors: tuple[str, ...], features: torch.Tensor
) -> list[str]:
    """Return a deterministic nested farthest-point order without using labels."""
    index = {policy_id: position for position, policy_id in enumerate(policy_ids)}
    pool = sorted(policy_ids)
    values = features[[index[policy_id] for policy_id in pool]].flatten(1).numpy()
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (values - mean) / scale
    by_id = {policy_id: normalized[position] for position, policy_id in enumerate(pool)}
    selected = list(anchors)
    remaining = [policy_id for policy_id in pool if policy_id not in selected]
    while remaining:
        scores = []
        for policy_id in remaining:
            minimum = min(np.linalg.norm(by_id[policy_id] - by_id[chosen]) for chosen in selected)
            scores.append((float(minimum), policy_id))
        _, chosen = max(scores, key=lambda item: (item[0], item[1]))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.array(values, dtype=float), q))


def main() -> None:
    results_dir = OUT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads((SOURCE / "policies/prefill_decode/manifest.json").read_text())
    extension_manifest = json.loads((OUT / "extension/policies/manifest.json").read_text())
    if len(extension_manifest) != 18:
        raise RuntimeError("expected exactly 18 calibration-only extension policies")
    manifest = [*base_manifest, *extension_manifest]
    split = json.loads((SOURCE / "policies/prefill_decode/coverage_holdout.json").read_text())
    train_ids = sorted(split["train"])
    holdout_ids = set(split["holdout"])
    if len(train_ids) != 54 or len(holdout_ids) != 18 or set(train_ids) & holdout_ids:
        raise RuntimeError("expected the frozen disjoint 54/18 quality split")
    if not set(ANCHORS) <= set(train_ids):
        raise RuntimeError("all five dense/uniform anchors must remain in every subset")

    nll_rows = [
        *read_csv(SOURCE / "nll/prefill_decode.csv"),
        *read_csv(OUT / "extension/nll/extension.csv"),
    ]
    labels = {row["policy_id"]: float(row["target_delta_nll"]) for row in nll_rows}
    elapsed = {row["policy_id"]: float(row["elapsed_seconds"]) for row in nll_rows}
    errors = quality.load_local_errors()
    policies = [json.loads(Path(row["path"]).read_text()) for row in manifest]
    policy_ids = [row["policy_id"] for row in manifest]
    x_raw = torch.stack([quality.policy_features(policy, errors) for policy in policies])
    y = torch.tensor([labels[policy_id] for policy_id in policy_ids], dtype=torch.float64)
    index = {policy_id: position for position, policy_id in enumerate(policy_ids)}
    holdout_indices = [index[policy_id] for policy_id in sorted(holdout_ids)]

    extension_ids = [row["policy_id"] for row in extension_manifest]
    candidate_ids = [*train_ids, *extension_ids]
    candidate_features = x_raw[[index[policy_id] for policy_id in candidate_ids]]
    coverage = coverage_order(candidate_ids, ANCHORS, candidate_features)

    orders: list[tuple[str, int, list[str]]] = [("coverage", 0, coverage)]
    nonanchors = [policy_id for policy_id in candidate_ids if policy_id not in ANCHORS]
    for seed in RANDOM_SEEDS:
        shuffled = list(nonanchors)
        generator = np.random.default_rng(seed)
        generator.shuffle(shuffled)
        orders.append(("random", seed, [*ANCHORS, *shuffled]))

    cache: dict[tuple[str, ...], dict[str, float]] = {}

    def fit(selected: list[str]) -> dict[str, float]:
        key = tuple(sorted(selected))
        if key in cache:
            return cache[key]
        train_mask = torch.tensor([policy_id in key for policy_id in policy_ids])
        feature_scale = x_raw[train_mask].sum((1, 2, 3, 4)).mean().clamp(min=1e-12)
        started = time.perf_counter()
        prediction, _ = quality.fit_variant("full_phase_aware", x_raw / feature_scale, y, train_mask)
        fit_seconds = time.perf_counter() - started
        result = quality.metrics(
            [float(y[position]) for position in holdout_indices],
            [float(prediction[position]) for position in holdout_indices],
        )
        result.update({"fit_seconds": fit_seconds, "feature_scale": float(feature_scale)})
        cache[key] = result
        return result

    rows: list[dict[str, object]] = []
    for sampling, seed, order in orders:
        for size in SIZES:
            selected = order[:size]
            result = fit(selected)
            rows.append({
                "sampling": sampling,
                "seed": seed,
                "calibration_policies": size,
                "selected_policy_ids": ";".join(selected),
                "holdout_policies": len(holdout_ids),
                "holdout_mae": result["mae"],
                "holdout_rmse": result["rmse"],
                "holdout_spearman": result["spearman"],
                "holdout_mean_signed_error": result["mean_signed_error"],
                "fit_seconds": result["fit_seconds"],
                "label_gpu_minutes": sum(elapsed[policy_id] for policy_id in selected) / 60.0,
                "feature_scale": result["feature_scale"],
            })

    with (results_dir / "learning_curve_repeats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for size in SIZES:
        coverage_row = next(
            row for row in rows if row["sampling"] == "coverage" and row["calibration_policies"] == size
        )
        random_rows = [
            row for row in rows if row["sampling"] == "random" and row["calibration_policies"] == size
        ]
        item: dict[str, object] = {
            "calibration_policies": size,
            "coverage_holdout_mae": coverage_row["holdout_mae"],
            "coverage_holdout_rmse": coverage_row["holdout_rmse"],
            "coverage_holdout_spearman": coverage_row["holdout_spearman"],
            "coverage_label_gpu_minutes": coverage_row["label_gpu_minutes"],
            "coverage_fit_seconds": coverage_row["fit_seconds"],
        }
        for field in ("holdout_mae", "holdout_rmse", "holdout_spearman", "label_gpu_minutes", "fit_seconds"):
            values = [float(row[field]) for row in random_rows]
            item[f"random_{field}_q25"] = quantile(values, 0.25)
            item[f"random_{field}_median"] = quantile(values, 0.50)
            item[f"random_{field}_q75"] = quantile(values, 0.75)
        summary_rows.append(item)

    with (results_dir / "learning_curve_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    full_mae = float(summary_rows[-1]["random_holdout_mae_median"])
    threshold = 1.05 * full_mae
    elbow = None
    for position, row in enumerate(summary_rows):
        if all(float(later["random_holdout_mae_median"]) <= threshold for later in summary_rows[position:]):
            elbow = int(row["calibration_policies"])
            break
    frozen_result = fit(train_ids)
    frozen_label_gpu_minutes = sum(elapsed[policy_id] for policy_id in train_ids) / 60.0
    frozen_rows = read_csv(ABLATION64 / "quality/metrics.csv")
    frozen = next(row for row in frozen_rows if row["variant"] == "full_phase_aware" and row["split"] == "holdout")
    validation = {
        "source": str(SOURCE),
        "sizes": list(SIZES),
        "anchors": list(ANCHORS),
        "random_seeds": list(RANDOM_SEEDS),
        "holdout_ids": sorted(holdout_ids),
        "extension_policy_ids": extension_ids,
        "candidate_policy_ids": candidate_ids,
        "frozen_main_policy_ids": train_ids,
        "elbow_rule": "smallest n whose random-median MAE is within 5% of n=72 and remains so",
        "observed_elbow": elbow,
        "full_mae": full_mae,
        "elbow_mae_threshold": threshold,
        "frozen_main_54_holdout_mae": frozen_result["mae"],
        "frozen_main_54_holdout_rmse": frozen_result["rmse"],
        "frozen_main_54_holdout_spearman": frozen_result["spearman"],
        "frozen_main_54_label_gpu_minutes": frozen_label_gpu_minutes,
        "selected_54_mae_abs_difference_vs_064": abs(float(frozen_result["mae"]) - float(frozen["mae"])),
        "selected_54_rmse_abs_difference_vs_064": abs(float(frozen_result["rmse"]) - float(frozen["rmse"])),
        "selected_54_spearman_abs_difference_vs_064": abs(float(frozen_result["spearman"]) - float(frozen["spearman"])),
    }
    validation["selected_54_matches_064_at_1e-5"] = max(
        validation["selected_54_mae_abs_difference_vs_064"],
        validation["selected_54_rmse_abs_difference_vs_064"],
        validation["selected_54_spearman_abs_difference_vs_064"],
    ) <= 1e-5
    (results_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if not validation["selected_54_matches_064_at_1e-5"]:
        raise RuntimeError("54-policy endpoint does not reproduce the frozen 064 quality model")


if __name__ == "__main__":
    main()
