#!/usr/bin/env python3
"""Combine representative and recovery task-quality summaries into all points."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    sources = (
        ROOT / "task_quality_continuous/summary.csv",
        ROOT / "task_quality_recovery/summary.csv",
    )
    rows = [row for source in sources for row in read(source)]
    expected = {(str(point), dataset) for point in range(12) for dataset in ("cnn_dm_1000", "dsum", "IWSLT")}
    keys = {(row["point"], row["dataset"]) for row in rows}
    if keys != expected or len(rows) != len(expected):
        raise RuntimeError(f"incomplete/duplicate combined task metrics: got={len(rows)} expected={len(expected)}")
    rows.sort(key=lambda row: (int(row["point"]), row["dataset"]))
    out = ROOT / "task_quality_all"; out.mkdir(exist_ok=True)
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(out / "summary.csv")


if __name__ == "__main__":
    main()
