#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common_sparse_bf16_proxy import (
    DEBUG_ROOT,
    FAKE_ROOT,
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
    parser = argparse.ArgumentParser(description="Generate controlled policy pairs matched on raw local error.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-014-root", type=Path, default=SOURCE_014_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--counts", default="32,64")
    parser.add_argument("--pairs-per-count", type=int, default=8)
    parser.add_argument("--candidates-per-count", type=int, default=20000)
    parser.add_argument("--raw-window-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    counts = [int(item) for item in args.counts.split(",") if item.strip()]
    out_root = args.output_root / "controlled"
    metadata: dict[str, Any] = {
        "metric": args.metric,
        "counts": counts,
        "pairs_per_count": args.pairs_per_count,
        "candidates_per_count": args.candidates_per_count,
        "raw_window_frac": args.raw_window_frac,
        "seed": args.seed,
        "methods": {},
    }
    for method in methods:
        rows, method_meta = build_method_policies(args, method, counts)
        path = out_root / "policies" / f"controlled_policies_{method}.csv"
        write_csv(path, rows)
        metadata["methods"][method] = method_meta | {"policies": len(rows), "path": str(path)}
        print(f"wrote {len(rows)} {method} policies to {path}")
    write_json(out_root / "policies" / "controlled_policy_metadata.json", metadata)


def build_method_policies(args: argparse.Namespace, method: str, counts: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed + method_seed(method))
    local_rows = load_local_rows(args, method)
    fit = load_final_fit(args.output_root, method)
    module_rows = [
        {
            "name": row["module_name"],
            "layer": int(f(row, "layer")),
            "type": row["module_type"],
            "error": f(row, args.metric),
        }
        for row in local_rows
    ]
    out: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"counts": {}}
    for count in counts:
        candidates = sample_candidates(module_rows, fit, count=count, n=args.candidates_per_count, rng=rng)
        pairs = select_pairs(candidates, pairs=args.pairs_per_count, raw_window_frac=args.raw_window_frac)
        meta["counts"][count] = {"candidate_count": len(candidates), "pair_count": len(pairs)}
        for pair_idx, (low, high) in enumerate(pairs):
            pair_id = f"{method}_c{count:03d}_pair{pair_idx:02d}"
            for arm, item in (("low_final", low), ("high_final", high)):
                out.append(
                    {
                        "policy_id": f"{pair_id}_{arm}",
                        "pair_id": pair_id,
                        "arm": arm,
                        "sample_kind": "controlled_raw_matched",
                        "selected_modules": count,
                        "selected_names": selected_to_text(item["names"]),
                        "raw_error_sum": item["raw"],
                        "final_proxy_sum": item["final"],
                        "raw_pair_gap": abs(high["raw"] - low["raw"]),
                        "final_pair_gap": high["final"] - low["final"],
                    }
                )
    return out, meta


def load_local_rows(args: argparse.Namespace, method: str) -> list[dict[str, Any]]:
    if method == "sparse_bf16":
        return load_sparse_local_errors(args.source_014_root)
    path = args.source_015_root / "sensitivity" / "module_method_kernel_local_errors.csv"
    rows = [row for row in read_csv(path) if row.get("method") == method]
    if not rows:
        raise RuntimeError(f"no local rows for {method}")
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_final_fit(output_root: Path, method: str) -> dict[str, Any]:
    path = output_root / "model" / f"fitted_{method}_proxy.json"
    return json.loads(path.read_text())


def sample_candidates(module_rows: list[dict[str, Any]], fit: dict[str, Any], *, count: int, n: int, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for _ in range(n):
        selected = rng.sample(module_rows, count)
        names = sorted(row["name"] for row in selected)
        raw = sum(row["error"] for row in selected)
        final = sum(row["error"] * final_coef(fit, row["layer"], row["type"]) for row in selected)
        out.append({"names": names, "raw": raw, "final": final})
    dedup = {selected_to_text(item["names"]): item for item in out}
    return sorted(dedup.values(), key=lambda item: item["raw"])


def final_coef(fit: dict[str, Any], layer: int, typ: str) -> float:
    return float(fit["layer_coef"][str(layer)]) * float(fit["type_coef"][typ])


def select_pairs(candidates: list[dict[str, Any]], *, pairs: int, raw_window_frac: float) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    scored = []
    raw_values = [item["raw"] for item in candidates]
    raw_span = max(raw_values) - min(raw_values)
    window = max(raw_span * raw_window_frac, 1e-12)
    for i, left in enumerate(candidates):
        j = i + 1
        best = None
        while j < len(candidates) and candidates[j]["raw"] - left["raw"] <= window:
            right = candidates[j]
            gap = abs(right["final"] - left["final"])
            if best is None or gap > best[0]:
                best = (gap, right)
            j += 1
        if best is None:
            continue
        right = best[1]
        low, high = (left, right) if left["final"] <= right["final"] else (right, left)
        scored.append((high["final"] - low["final"], low, high))
    selected = []
    used = set()
    for _, low, high in sorted(scored, key=lambda item: item[0], reverse=True):
        low_key = selected_to_text(low["names"])
        high_key = selected_to_text(high["names"])
        if low_key in used or high_key in used:
            continue
        selected.append((low, high))
        used.update([low_key, high_key])
        if len(selected) >= pairs:
            break
    if len(selected) < pairs:
        raise RuntimeError(f"only found {len(selected)} pairs, requested {pairs}")
    return selected


def method_seed(method: str) -> int:
    return sum(ord(ch) for ch in method)


if __name__ == "__main__":
    main()
