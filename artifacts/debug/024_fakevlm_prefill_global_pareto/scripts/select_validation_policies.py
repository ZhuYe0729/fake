#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common_fakevlm_pareto import DEBUG_ROOT, f, parse_batches, pareto_policy_path, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative FakeVLM Pareto points for real validation.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--batches", default="all")
    parser.add_argument("--points-per-batch", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for batch in parse_batches(args.batches):
        frontier = read_csv(args.output_root / "pareto" / f"batch_{batch}" / "pareto_unique_points.csv")
        selected = select_even(frontier, args.points_per_batch)
        for row in selected:
            point = int(f(row, "point_index"))
            budget = f(row, "quality_budget")
            path = find_policy(args.output_root, batch, point, budget)
            rows.append({**row, "policy_json": str(path), "selection_reason": "even_quality_grid"})
    write_csv(args.output_root / "validation" / "selected_pareto_points.csv", rows)
    write_json(args.output_root / "validation" / "selected_pareto_points.json", rows)
    print(f"selected {len(rows)} policies")


def select_even(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    if count <= 1:
        return [rows[0]]
    indices = sorted({round(i * (len(rows) - 1) / (count - 1)) for i in range(count)})
    return [rows[i] for i in indices]


def find_policy(output_root: Path, batch: int, point: int, budget: float) -> Path:
    exact = pareto_policy_path(output_root, batch, point, budget)
    if exact.exists():
        return exact
    matches = sorted((output_root / "pareto" / f"batch_{batch}" / "policies").glob(f"point_{point:03d}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no policy for batch={batch} point={point}")
    return matches[0]


if __name__ == "__main__":
    main()
