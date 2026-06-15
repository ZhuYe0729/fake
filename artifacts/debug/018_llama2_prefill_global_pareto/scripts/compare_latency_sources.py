#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from build_cost_table import load_latency_rows, latency_map
from common_pareto import DEBUG_ROOT, METHODS, f, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fresh prefill microbench against existing oracle-summary latency.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fresh = latency_map(load_latency_rows(args.output_root, "fresh"))
    existing = latency_map(load_latency_rows(args.output_root, "existing"))
    rows = []
    for key in sorted(existing):
        erow = existing[key]
        frow = fresh.get(key)
        method, group, n, k = key
        if frow is None:
            rows.append(
                {
                    "method": method,
                    "linear_group": group,
                    "n": n,
                    "k": k,
                    "existing_prefill_ms": f(erow, "prefill_ms"),
                    "fresh_prefill_ms": "",
                    "delta_ms": "",
                    "relative_delta": "",
                    "status": "missing_fresh",
                }
            )
            continue
        existing_ms = f(erow, "prefill_ms")
        fresh_ms = f(frow, "prefill_ms")
        rows.append(
            {
                "method": method,
                "linear_group": group,
                "n": n,
                "k": k,
                "existing_prefill_ms": existing_ms,
                "fresh_prefill_ms": fresh_ms,
                "delta_ms": fresh_ms - existing_ms,
                "relative_delta": (fresh_ms - existing_ms) / existing_ms if existing_ms else 0.0,
                "status": "ok",
            }
        )
    write_csv(args.output_root / "latency" / "latency_source_comparison.csv", rows)
    write_json(
        args.output_root / "latency" / "latency_source_comparison_metadata.json",
        {
            "rows": len(rows),
            "methods": list(METHODS),
            "fresh_rows": len(fresh),
            "existing_rows": len(existing),
        },
    )
    print(f"wrote {len(rows)} latency comparison rows")


if __name__ == "__main__":
    main()
