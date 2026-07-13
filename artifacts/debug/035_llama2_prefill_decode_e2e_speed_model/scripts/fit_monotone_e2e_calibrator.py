#!/usr/bin/env python3
"""Fit and leave-one-out validate a monotone correction of linear kernel latency."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


def elapsed(runs: Path, output: int) -> float:
    files = sorted(runs.glob(f"measured_*_o{output}.json"))
    return statistics.median(json.loads(path.read_text())["elapsed_ms"] for path in files)


def pava(values: list[float]) -> list[float]:
    """Non-decreasing isotonic regression for observations already sorted by x."""
    blocks: list[list[float]] = []  # [sum, count]
    for value in values:
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            blocks[-1][0] += right[0]
            blocks[-1][1] += right[1]
    return [total / count for total, count in blocks for _ in range(int(count))]


def predict(train: list[tuple[float, float]], x: float) -> float:
    ordered = sorted(train)
    xs = [item[0] for item in ordered]
    ys = pava([item[1] for item in ordered])
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for index in range(1, len(xs)):
        if x <= xs[index]:
            weight = (x - xs[index - 1]) / (xs[index] - xs[index - 1])
            return ys[index - 1] + weight * (ys[index] - ys[index - 1])
    raise AssertionError("unreachable")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root.parent / "034_llama2_7b_chat_wikitext_pareto_solver"
    points = list(csv.DictReader((source / "prefill_decode/pareto/pareto_points.csv").open()))
    by_point = {int(row["point_index"]): row for row in points}
    measured: dict[int, tuple[float, int]] = {}
    # Existing formal 10-repeat anchors.
    for point in (0, 3, 6, 11):
        runs = source / "validation/prefill_decode/speed_official" / f"point_{point}" / "runs"
        measured[point] = (elapsed(runs, 80), len(list(runs.glob("measured_*_o80.json"))))
    # New three-repeat calibration points.
    for directory in sorted((root / "measurements").glob("point_*")):
        runs = directory / "runs"
        files = list(runs.glob("measured_*_o80.json"))
        if files:
            point = int(directory.name.split("_")[1])
            measured[point] = (elapsed(runs, 80), len(files))

    rows = []
    for point, (e2e, samples) in sorted(measured.items()):
        raw = float(by_point[point]["raw_predicted_linear_ms"])
        rows.append({"point_index": point, "samples": samples, "raw_linear_ms": raw, "formal_e2e_ms": e2e})
    pairs = [(float(row["raw_linear_ms"]), float(row["formal_e2e_ms"])) for row in rows]
    for index, row in enumerate(rows):
        row["monotone_fit_ms"] = predict(pairs, float(row["raw_linear_ms"]))
        loo = pairs[:index] + pairs[index + 1:]
        row["loo_monotone_ms"] = predict(loo, float(row["raw_linear_ms"])) if len(loo) >= 2 else ""
    out = root / "e2e_calibration.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    loo_rows = [row for row in rows if row["loo_monotone_ms"] != ""]
    raw_scale = float(rows[0]["formal_e2e_ms"]) / float(rows[0]["raw_linear_ms"])
    raw_mae = statistics.mean(abs(float(row["formal_e2e_ms"]) - raw_scale * float(row["raw_linear_ms"])) for row in loo_rows)
    loo_mae = statistics.mean(abs(float(row["formal_e2e_ms"]) - float(row["loo_monotone_ms"])) for row in loo_rows)
    metrics = {"points": len(rows), "raw_single_anchor_mae_ms": raw_mae,
               "loo_monotone_mae_ms": loo_mae,
               "feasibility": {"7": "oom_formal_0.9", "8": "oom_formal_0.9",
                               "9": "oom_formal_0.9"},
               "status": "debug_only_not_yet_used_for_pareto"}
    (root / "e2e_calibration_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
