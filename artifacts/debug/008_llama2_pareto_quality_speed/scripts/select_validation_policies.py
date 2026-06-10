#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common_pareto import DEBUG_ROOT, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative Pareto policies for real validation.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--points", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.output_root / "pareto" / "pareto_unique_points.csv")
    if not rows:
        raise RuntimeError("no unique pareto points")
    indices = representative_indices(len(rows), args.points)
    selected = []
    for rank, idx in enumerate(indices):
        row = dict(rows[idx])
        row["validation_rank"] = rank
        row["source_unique_index"] = idx
        selected.append(row)
    write_csv(args.output_root / "validation" / "selected_pareto_points.csv", selected)
    write_json(
        args.output_root / "validation" / "selected_pareto_points.json",
        {
            "points": selected,
            "note": "Use matching point_index policy JSON/CSV under pareto/policies for validation.",
        },
    )
    print(f"selected {len(selected)} validation points")


def representative_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [0]
    raw = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    out = []
    for idx in raw:
        if idx not in out:
            out.append(idx)
    return out


if __name__ == "__main__":
    main()
