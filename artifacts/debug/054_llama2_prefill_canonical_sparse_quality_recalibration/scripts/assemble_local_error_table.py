#!/usr/bin/env python3
"""Replace legacy sparse feature rows with wrapper-measured canonical rows."""
from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"))
LEGACY = ROOT / "artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    sparse = {method: read(EXPERIMENT / f"local_errors/{method}.csv")
              for method in ("sparse_bf16", "sparse_nvfp4")}
    rows = [row for row in read(LEGACY) if row["method"] not in sparse]
    for method, values in sparse.items():
        rows.extend(values)
    output = EXPERIMENT / "local_errors/module_method_errors.csv"
    fields = sorted({field for row in rows for field in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
