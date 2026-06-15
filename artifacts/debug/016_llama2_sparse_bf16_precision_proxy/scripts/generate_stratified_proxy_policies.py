#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from common_sparse_bf16_proxy import (
    DEBUG_ROOT,
    FAKE_ROOT,
    LAYERS,
    LINEAR_TYPES,
    LOCAL_ERROR_METRIC,
    SOURCE_014_ROOT,
    f,
    load_sparse_local_errors,
    selected_to_text,
    write_csv,
    write_json,
)


SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"
METHODS = ("sparse_bf16", "dense_nvfp4", "sparse_nvfp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw-bin stratified policies with layer/type composition diversity.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-014-root", type=Path, default=SOURCE_014_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--counts", default="16,32,64,112")
    parser.add_argument("--raw-bins", type=int, default=4)
    parser.add_argument("--policies-per-bin", type=int, default=5)
    parser.add_argument("--candidates-per-count", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    counts = [int(item) for item in args.counts.split(",") if item.strip()]
    metadata: dict[str, Any] = {
        "metric": args.metric,
        "counts": counts,
        "raw_bins": args.raw_bins,
        "policies_per_bin": args.policies_per_bin,
        "candidates_per_count": args.candidates_per_count,
        "seed": args.seed,
        "methods": {},
    }
    for method in methods:
        rows, meta = build_method(args, method, counts)
        path = args.output_root / "stratified" / "policies" / f"stratified_policies_{method}.csv"
        write_csv(path, rows)
        metadata["methods"][method] = meta | {"policies": len(rows), "path": str(path)}
        print(f"wrote {len(rows)} {method} policies to {path}")
    write_json(args.output_root / "stratified" / "policies" / "stratified_policy_metadata.json", metadata)


def build_method(args: argparse.Namespace, method: str, counts: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed + method_seed(method))
    modules = load_modules(args, method)
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"counts": {}}
    for count in counts:
        candidates = sample_candidates(modules, count=count, n=args.candidates_per_count, rng=rng)
        bin_items = split_raw_bins(candidates, bins=args.raw_bins)
        count_meta = {"candidate_count": len(candidates), "bins": []}
        for bin_idx, items in enumerate(bin_items):
            selected = select_diverse(items, k=args.policies_per_bin)
            count_meta["bins"].append(
                {
                    "bin": bin_idx,
                    "candidates": len(items),
                    "selected": len(selected),
                    "raw_min": min(item["raw"] for item in items),
                    "raw_max": max(item["raw"] for item in items),
                }
            )
            for item_idx, item in enumerate(selected):
                policy_id = f"{method}_strat_c{count:03d}_b{bin_idx:02d}_{item_idx:02d}"
                rows.append(
                    {
                        "policy_id": policy_id,
                        "sample_kind": "stratified_raw_bin_diverse",
                        "method": method,
                        "selected_modules": count,
                        "raw_bin": bin_idx,
                        "selected_names": selected_to_text(item["names"]),
                        "raw_error_sum": item["raw"],
                        "layer_entropy": item["layer_entropy"],
                        "type_entropy": item["type_entropy"],
                    }
                )
        meta["counts"][count] = count_meta
    return rows, meta


def load_modules(args: argparse.Namespace, method: str) -> list[dict[str, Any]]:
    if method == "sparse_bf16":
        rows = load_sparse_local_errors(args.source_014_root)
    else:
        path = args.source_015_root / "sensitivity" / "module_method_kernel_local_errors.csv"
        rows = [row for row in read_csv(path) if row.get("method") == method]
    if not rows:
        raise RuntimeError(f"no local rows for {method}")
    return [
        {
            "name": row["module_name"],
            "layer": int(f(row, "layer")),
            "type": row["module_type"],
            "error": f(row, args.metric),
        }
        for row in rows
    ]


def read_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def sample_candidates(modules: list[dict[str, Any]], *, count: int, n: int, rng: random.Random) -> list[dict[str, Any]]:
    out = {}
    for _ in range(n):
        selected = rng.sample(modules, count)
        names = sorted(row["name"] for row in selected)
        key = selected_to_text(names)
        if key in out:
            continue
        raw = sum(row["error"] for row in selected)
        layer_counts = np.zeros(len(LAYERS), dtype=np.float64)
        type_counts = np.zeros(len(LINEAR_TYPES), dtype=np.float64)
        for row in selected:
            layer_counts[LAYERS.index(row["layer"])] += 1.0
            type_counts[LINEAR_TYPES.index(row["type"])] += 1.0
        composition = np.r_[layer_counts / count, type_counts / count]
        out[key] = {
            "names": names,
            "raw": raw,
            "composition": composition,
            "layer_entropy": entropy(layer_counts),
            "type_entropy": entropy(type_counts),
        }
    return sorted(out.values(), key=lambda item: item["raw"])


def split_raw_bins(candidates: list[dict[str, Any]], *, bins: int) -> list[list[dict[str, Any]]]:
    if len(candidates) < bins:
        raise RuntimeError(f"not enough candidates={len(candidates)} for bins={bins}")
    chunks = []
    for idx in range(bins):
        start = round(idx * len(candidates) / bins)
        end = round((idx + 1) * len(candidates) / bins)
        chunks.append(candidates[start:end])
    return chunks


def select_diverse(items: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    if len(items) < k:
        raise RuntimeError(f"only {len(items)} candidates in bin, requested {k}")
    center = np.mean([item["composition"] for item in items], axis=0)
    first = max(items, key=lambda item: float(np.linalg.norm(item["composition"] - center)))
    selected = [first]
    remaining = [item for item in items if item is not first]
    while len(selected) < k:
        best = max(
            remaining,
            key=lambda item: min(float(np.linalg.norm(item["composition"] - chosen["composition"])) for chosen in selected),
        )
        selected.append(best)
        remaining.remove(best)
    return sorted(selected, key=lambda item: item["raw"])


def entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def method_seed(method: str) -> int:
    return sum(ord(ch) for ch in method)


if __name__ == "__main__":
    main()
