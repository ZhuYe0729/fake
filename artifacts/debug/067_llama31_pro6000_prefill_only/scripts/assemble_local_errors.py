#!/usr/bin/env python3
"""Combine copied non-sparse features with freshly measured canonical sparse rows."""
from __future__ import annotations

import csv
from collections import Counter

from common import INPUTS, RUN, sha256, write_json


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = read(INPUTS / "local_errors/non_sparse_module_method_errors.csv")
    rows = [row for row in source if row["method"] not in {"sparse_bf16", "sparse_nvfp4"}]
    for method in ("sparse_bf16", "sparse_nvfp4"):
        fresh = read(RUN / f"local_errors/{method}.csv")
        if len(fresh) != 224:
            raise RuntimeError(f"{method}: expected 224 rows, found {len(fresh)}")
        rows.extend(fresh)
    output = RUN / "local_errors/module_method_errors.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    write_json(RUN / "local_errors/provenance.json", {
        "frozen_non_sparse_input": str(INPUTS / "local_errors/non_sparse_module_method_errors.csv"),
        "frozen_non_sparse_input_sha256": sha256(INPUTS / "local_errors/non_sparse_module_method_errors.csv"),
        "fresh_sparse_bf16_sha256": sha256(RUN / "local_errors/sparse_bf16.csv"),
        "fresh_sparse_nvfp4_sha256": sha256(RUN / "local_errors/sparse_nvfp4.csv"),
        "output_sha256": sha256(output),
        "rows_by_method": dict(Counter(row["method"] for row in rows)),
    })
    print(output)


if __name__ == "__main__":
    main()
