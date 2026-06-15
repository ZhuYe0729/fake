#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Any

from common_pareto import DEBUG_ROOT, f, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join Pareto E2E and quality validation results.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    e2e_path = args.output_root / "validation" / "pareto_e2e_validation.csv"
    quality_path = args.output_root / "validation" / "pareto_quality_validation.csv"
    e2e_rows = read_csv(e2e_path) if e2e_path.exists() else []
    quality_rows = read_csv(quality_path) if quality_path.exists() else []
    quality_by_point = {int(f(row, "point_index")): row for row in quality_rows}

    dense_e2e = next(
        (f(row, "e2e_total_ms") for row in e2e_rows if int(f(row, "point_index")) == 0 and row.get("e2e_status") == "ok"),
        None,
    )
    joined = []
    for row in e2e_rows:
        point = int(f(row, "point_index"))
        qrow = quality_by_point.get(point, {})
        item = dict(row)
        e2e_total = f(row, "e2e_total_ms")
        item.update(
            {
                "nll": qrow.get("nll", ""),
                "arc_acc": qrow.get("arc_acc", ""),
                "arc_acc_norm": qrow.get("arc_acc_norm", ""),
                "e2e_speedup_vs_dense": dense_e2e / e2e_total if dense_e2e and e2e_total > 0 else "",
                "predicted_to_e2e_ratio": (
                    f(row, "predicted_total_latency_ms") / e2e_total if e2e_total > 0 else ""
                ),
            }
        )
        joined.append(item)

    write_csv(args.output_root / "validation" / "pareto_validation_joined.csv", joined)
    correlations = validation_correlations(joined, e2e_rows)
    write_csv(args.output_root / "validation" / "validation_correlations.csv", correlations)
    write_json(
        args.output_root / "validation" / "validation_summary_metadata.json",
        {
            "e2e_rows": len(e2e_rows),
            "quality_rows": len(quality_rows),
            "joined_rows": len(joined),
        },
    )
    print(f"wrote {len(joined)} joined validation rows")


def validation_correlations(rows: list[dict[str, Any]], e2e_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    ok_rows = [r for r in rows if r.get("e2e_status") == "ok"]

    pairs = [(f(row, "quality_cost"), f(row, "e2e_total_ms")) for row in ok_rows if row.get("e2e_total_ms", "") != ""]
    if len(pairs) >= 3:
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        out.append(
            {
                "x": "quality_cost",
                "y": "e2e_total_ms",
                "rows": len(pairs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )

    pairs = [(f(row, "predicted_total_latency_ms"), f(row, "e2e_total_ms")) for row in ok_rows if row.get("e2e_total_ms", "") != ""]
    if len(pairs) >= 3:
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        out.append(
            {
                "x": "predicted_total_latency_ms",
                "y": "e2e_total_ms",
                "rows": len(pairs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )

    pairs = [(f(row, "predicted_total_latency_ms"), f(row, "e2e_prefill_ms")) for row in ok_rows if row.get("e2e_prefill_ms", "") != ""]
    if len(pairs) >= 3:
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        out.append(
            {
                "x": "predicted_total_latency_ms",
                "y": "e2e_prefill_ms",
                "rows": len(pairs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )

    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


if __name__ == "__main__":
    main()
