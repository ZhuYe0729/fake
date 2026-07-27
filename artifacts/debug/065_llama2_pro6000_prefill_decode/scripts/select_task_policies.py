#!/usr/bin/env python3
"""Select measured-frontier policies for full downstream evaluation."""
from __future__ import annotations

import csv
import json
import math

from common import RUN, write_json


def main() -> None:
    rows = []
    with (RUN / "pareto/predicted_points.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["policy_id"]
            nll = json.loads((RUN / "closure" / label / "nll.json").read_text())
            speed = json.loads((RUN / "closure" / label / "speed/summary.json").read_text())
            rows.append({"label": label, "policy": str(RUN / "pareto/policies" / f"{label}.json"),
                         "delta_nll": nll["avg_nll"], "median_ms": speed["median_ms"]})
    dense_nll = json.loads((RUN / "closure/uniform_p00/nll.json").read_text())["avg_nll"]
    for row in rows:
        row["delta_nll"] -= dense_nll
    frontier = []
    for row in sorted(rows, key=lambda item: (item["delta_nll"], item["median_ms"])):
        if not frontier or row["median_ms"] < min(item["median_ms"] for item in frontier):
            frontier.append(row)
    chosen = {min(frontier, key=lambda row: abs(row["delta_nll"] - target))["label"]
              for target in (0.01, 0.03, 0.05, 0.15)}
    chosen.add(min(frontier, key=lambda row: row["median_ms"])["label"])
    if len(frontier) > 2:
        x0, x1 = frontier[0]["delta_nll"], frontier[-1]["delta_nll"]
        y0, y1 = frontier[0]["median_ms"], frontier[-1]["median_ms"]
        def distance(row):
            x = (row["delta_nll"] - x0) / max(x1 - x0, 1e-12)
            y = (row["median_ms"] - y0) / max(abs(y1 - y0), 1e-12)
            # With latency improving downwards, normalized endpoints are
            # (0, 0) and (1, -1), so their chord is x + y = 0.
            return abs(y + x) / math.sqrt(2.0)
        chosen.add(max(frontier[1:-1], key=distance)["label"])
    selected = [f"uniform_p{index:02d}" for index in range(5)] + sorted(chosen)
    write_json(RUN / "tasks/selection.json", {"selection_rule": "uniform p00-p04; nearest measured delta-NLL targets 0.01/0.03/0.05/0.15; normalized-curve knee; max speed", "selected": selected, "frontier": frontier})
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
