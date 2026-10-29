#!/usr/bin/env python3
"""Merge PMPD answer shards, rejecting overlap and missing question ranges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    rows: dict[int, dict] = {}
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = int(row["question_id"])
            if question_id in rows:
                raise ValueError(f"duplicate question_id={question_id}: {path}")
            rows[question_id] = row
    ordered = [rows[key] for key in sorted(rows)]
    if args.expected_count is not None and len(ordered) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} rows, found {len(ordered)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(ordered),
                      "first_question_id": min(rows), "last_question_id": max(rows)}, indent=2))


if __name__ == "__main__":
    main()
