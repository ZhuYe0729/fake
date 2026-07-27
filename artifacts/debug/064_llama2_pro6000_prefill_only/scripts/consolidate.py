#!/usr/bin/env python3
"""Build the final measured table and metric Pareto figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from common import PROTOCOL, RESULTS, RUN, sha256, write_json

TASK_METRICS = {
    "wikitext": "word_perplexity,",
    "winogrande": "acc,",
    "arc_easy": "acc,",
    "arc_challenge": "acc_norm,",
    "mmlu": "acc,",
}


def metric(label: str, task: str):
    path = RUN / "pareto/validation/tasks" / label / task / "full/result.json"
    if not path.exists():
        return ""
    values = json.loads(path.read_text())["metrics"]
    prefix = TASK_METRICS[task]
    matched = [value for key, value in values.items() if key.startswith(prefix)]
    return matched[0] if matched else ""


def main() -> None:
    predicted = {}
    pareto_csv = RUN / "pareto/pareto_points.csv"
    if pareto_csv.exists():
        with pareto_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                predicted[f"point_{int(row['point_index']):03d}"] = row
    labels = [f"uniform_p{index:02d}" for index in range(5)] + sorted(predicted)
    dense_nll = json.loads((RUN / "closure/uniform_p00/nll.json").read_text())["avg_nll"]
    dense_ms = json.loads((RUN / "closure/uniform_p00/speed/summary.json").read_text())["median_ms"]
    rows = []
    for label in labels:
        closure = RUN / "closure" / label
        if not (closure / "nll.json").is_file() or not (closure / "speed/summary.json").is_file():
            continue
        nll = json.loads((closure / "nll.json").read_text())
        speed = json.loads((closure / "speed/summary.json").read_text())
        policy_path = (RUN / "policies/prefill_only" / f"{label.removeprefix('uniform_')}.json"
                       if label.startswith("uniform_") else Path(predicted[label]["policy_json"]))
        raw = json.loads((closure / "speed/raw/measured_0.json").read_text())
        row = {"family": "uniform" if label.startswith("uniform") else "ours", "policy": label,
               "policy_json": str(policy_path), "policy_sha256": sha256(policy_path),
               "median_ms": speed["median_ms"], "measured_speedup": dense_ms / speed["median_ms"],
               "avg_nll": nll["avg_nll"], "actual_delta_nll": nll["avg_nll"] - dense_nll,
               "speed_cv": speed["cv"], "speed_gpu_uuid": raw.get("cuda_device_uuid"),
               "speed_measured_runs": len(speed["measured_elapsed_ms"]),
               "predicted_delta_nll": predicted.get(label, {}).get("predicted_delta_nll", ""),
               "raw_predicted_linear_ms": predicted.get(label, {}).get("raw_predicted_linear_ms", "")}
        row.update({task: metric(label, task) for task in TASK_METRICS})
        rows.append(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "complete_results.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figures = RESULTS / "figures"; figures.mkdir(exist_ok=True)
    for field in ("actual_delta_nll", *TASK_METRICS):
        usable = [row for row in rows if row[field] != ""]
        if not usable:
            continue
        for family, marker, color in (("uniform", "s", "#e45756"), ("ours", "o", "#4c78a8")):
            subset = [row for row in usable if row["family"] == family]
            plt.scatter([row["measured_speedup"] for row in subset], [float(row[field]) for row in subset],
                        label=family, marker=marker, color=color)
        plt.xlabel("Measured E2E speedup vs dense BF16")
        plt.ylabel(field)
        plt.legend(); plt.tight_layout(); plt.savefig(figures / f"pareto_speed_vs_{field}.png", dpi=180); plt.close()
    write_json(RESULTS / "source_summary.json", {"run_root": str(RUN), "rows": len(rows),
               "protocol": PROTOCOL, "smoke_results_excluded": True,
               "speed_source": "closure 1 warmup + 5 measured fresh-process generate-only runs",
               "quality_source": "closure 100-block direct vLLM prompt logprob plus full lm-eval tasks",
               "dense_reference_ms": dense_ms, "dense_reference_nll": dense_nll,
               "table": str(output), "figures": str(figures)})
    print(output)


if __name__ == "__main__":
    main()
