#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common_sparse_bf16_proxy import (
    DEBUG_ROOT,
    POLICY_COUNTS,
    POLICIES_PER_COUNT,
    SOURCE_014_ROOT,
    build_policy_rows,
    load_sparse_local_errors,
    policy_paths,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sampled sparse BF16 policies for proxy fitting.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-014-root", type=Path, default=SOURCE_014_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies-per-count", type=int, default=POLICIES_PER_COUNT)
    parser.add_argument("--counts", default=",".join(str(value) for value in POLICY_COUNTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = tuple(int(item) for item in args.counts.split(",") if item.strip())
    local_rows = load_sparse_local_errors(args.source_014_root)
    rows = build_policy_rows(local_rows, seed=args.seed, policies_per_count=args.policies_per_count, counts=counts)
    paths = policy_paths(args.output_root)
    write_csv(paths["policies"], rows)
    write_json(
        paths["policies"].with_suffix(".metadata.json"),
        {
            "source_014_root": str(args.source_014_root),
            "method": "sparse_bf16",
            "local_error_metric": "output_rel_mse",
            "seed": args.seed,
            "counts": counts,
            "policies_per_count": args.policies_per_count,
            "policies": len(rows),
        },
    )
    print(f"wrote {len(rows)} policies to {paths['policies']}")


if __name__ == "__main__":
    main()
