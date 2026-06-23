#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common_search_audit import (
    DEBUG_ROOT,
    DEFAULT_SUBSET_FRACTION,
    DEFAULT_SUBSET_SEED,
    DEFAULT_TEST_JSON,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed-random FakeClue subset manifest.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--fraction", type=float, default=DEFAULT_SUBSET_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.test_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    count = int(round(total * args.fraction))
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(total), count))
    rows = []
    for subset_index, source_index in enumerate(indices):
        item = data[source_index]
        rows.append(
            {
                "subset_index": subset_index,
                "source_index": source_index,
                "image": item.get("image", ""),
                "label": item.get("label", ""),
            }
        )
    write_csv(args.output_root / "subset" / "subset_indices.csv", rows)
    write_json(
        args.output_root / "subset" / "subset_manifest.json",
        {
            "test_json_file": args.test_json_file,
            "seed": args.seed,
            "fraction": args.fraction,
            "total_samples": total,
            "subset_samples": count,
            "indices_csv": args.output_root / "subset" / "subset_indices.csv",
        },
    )
    print(f"wrote subset samples={count}/{total} seed={args.seed}")


if __name__ == "__main__":
    main()
