#!/usr/bin/env python3
"""Build the complete measured prefill-decode table and paper figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from common import RESULTS, RUN, write_json


def main() -> None:
    labels = [f"uniform_p{index:02d}" for index in range(5)]
    with (RUN / "pareto/predicted_points.csv").open(newline="") as handle:
        labels.extend(row["policy_id"] for row in csv.DictReader(handle))
    labels = list(dict.fromkeys(labels))
    dense_nll = json.loads((RUN / "closure/uniform_p00/nll.json").read_text())["avg_nll"]
    dense_ms = json.loads((RUN / "closure/uniform_p00/speed/summary.json").read_text())["median_ms"]
    task_rows = {(row["label"], row["dataset"]): row
                 for row in json.loads((RUN / "tasks/summary.json").read_text()).get("rows", [])}
    rows = []
    for label in labels:
        root = RUN / "closure" / label
        nll = json.loads((root / "nll.json").read_text())
        speed = json.loads((root / "speed/summary.json").read_text())
        row = {"label": label, "family": "uniform" if label.startswith("uniform_") else "ours",
               "avg_nll": nll["avg_nll"], "actual_delta_nll": nll["avg_nll"] - dense_nll,
               "median_ms": speed["median_ms"], "measured_speedup_vs_dense": dense_ms / speed["median_ms"]}
        for dataset in ("cnn_dm_1000", "dsum", "IWSLT"):
            metrics = task_rows.get((label, dataset), {})
            for key in ("rougeL_percent", "bert_score_percent", "sacre_bleu"):
                row[f"{dataset}_{key}"] = metrics.get(key, "")
        rows.append(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "complete_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figures = RESULTS / "figures"; figures.mkdir(parents=True, exist_ok=True)
    plots = [
        ("actual_delta_nll", "Measured ΔNLL", "speed_vs_actual_delta_nll.png"),
        ("cnn_dm_1000_rougeL_percent", "CNN/DM ROUGE-L (%)", "speed_vs_cnn_rougel.png"),
        ("cnn_dm_1000_bert_score_percent", "CNN/DM BERTScore (%)", "speed_vs_cnn_bertscore.png"),
        ("dsum_rougeL_percent", "DialogSum ROUGE-L (%)", "speed_vs_dialogsum_rougel.png"),
        ("dsum_bert_score_percent", "DialogSum BERTScore (%)", "speed_vs_dialogsum_bertscore.png"),
        ("IWSLT_sacre_bleu", "IWSLT SacreBLEU", "speed_vs_iwslt_bleu.png"),
        ("IWSLT_rougeL_percent", "IWSLT ROUGE-L (%)", "speed_vs_iwslt_rougel.png"),
    ]
    for field, ylabel, filename in plots:
        available = [row for row in rows if row[field] != ""]
        if not available: continue
        for family, marker, color in (("uniform", "s", "#d62728"), ("ours", "o", "#202b3c")):
            subset = [row for row in available if row["family"] == family]
            if subset:
                plt.scatter([row["measured_speedup_vs_dense"] for row in subset],
                            [float(row[field]) for row in subset], marker=marker, color=color, label=family)
        plt.xlabel("Measured speedup vs dense BF16"); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend()
        plt.tight_layout(); plt.savefig(figures / filename, dpi=180); plt.close()
    write_json(RESULTS / "source_summary.json", {
        "scenario": "prefill_decode", "speed_source": "exclusive 1+5 measured vLLM E2E median",
        "quality_source": "100-block B8/O64 teacher-forced decode NLL",
        "task_source": "fixed PMPD generation", "rows": len(rows),
        "selected_task_points": json.loads((RUN / "tasks/selection.json").read_text())["selected"]})
    print(json.dumps({"rows": len(rows), "figures": len(list(figures.glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
