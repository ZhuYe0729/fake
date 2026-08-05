#!/usr/bin/env python3
"""Merge 061, 068, and final uniform measurements into the 068 report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from scenario import EXP, SOURCE_060, TARGET_ARC_CHALLENGE, TARGET_SPEEDUP


ANCHOR_MS = 1087.526896968484
BASE_NLL = 1.983390648407158
FIELDS = (
    "family",
    "policy",
    "source",
    "speed_ms",
    "speedup",
    "delta_nll",
    "wikitext",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
)
METRICS = {
    "wikitext": ("WikiText word perplexity", False),
    "winogrande": ("WinoGrande accuracy", True),
    "arc_easy": ("ARC-Easy accuracy", True),
    "arc_challenge": ("ARC-Challenge normalized accuracy", True),
    "mmlu": ("MMLU accuracy", True),
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return None if value in (None, "") else float(value)


def metric(path: Path, key: str):
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return number(payload["metrics"].get(key))


def candidate(policy: str) -> dict[str, object]:
    speed_path = EXP / "speed" / f"{policy}.json"
    nll_path = EXP / "nll" / f"{policy}.json"
    speed = json.loads(speed_path.read_text()) if speed_path.exists() else {}
    nll = json.loads(nll_path.read_text()) if nll_path.exists() else {}
    root = EXP / "task_quality/results" / policy
    return {
        "family": "ours",
        "policy": policy,
        "source": "068",
        "speed_ms": speed.get("median_ms"),
        "speedup": ANCHOR_MS / speed["median_ms"] if speed else None,
        "delta_nll": nll.get("avg_nll", 0.0) - BASE_NLL if nll else None,
        "wikitext": metric(root / "wikitext/full/result.json", "word_perplexity,none"),
        "winogrande": metric(root / "winogrande/full/result.json", "acc,none"),
        "arc_easy": metric(root / "arc_easy/full/result.json", "acc,none"),
        "arc_challenge": metric(root / "arc_challenge/full/result.json", "acc_norm,none"),
        "mmlu": metric(root / "mmlu/full/result.json", "acc,none"),
    }


def frontier(rows: list[dict[str, object]], metric_name: str) -> list[dict[str, object]]:
    valid = [row for row in rows if row["speedup"] is not None and row[metric_name] is not None]
    higher_is_better = METRICS[metric_name][1]
    retained = []
    for row in valid:
        dominated = any(
            float(other["speedup"]) >= float(row["speedup"])
            and (
                float(other[metric_name]) >= float(row[metric_name])
                if higher_is_better
                else float(other[metric_name]) <= float(row[metric_name])
            )
            and (
                float(other["speedup"]) > float(row["speedup"])
                or (
                    float(other[metric_name]) > float(row[metric_name])
                    if higher_is_better
                    else float(other[metric_name]) < float(row[metric_name])
                )
            )
            for other in valid
            if other is not row
        )
        if not dominated:
            retained.append(row)
    return sorted(retained, key=lambda row: float(row["speedup"]))


def main() -> None:
    rows = []
    for row in read(SOURCE_060 / "data/complete_results.csv"):
        rows.append({
            "family": row["family"],
            "policy": row["policy"],
            "source": "061/060",
            "speed_ms": number(row["speed_ms"]),
            "speedup": number(row["speedup"]),
            "delta_nll": None,
            "wikitext": number(row["wikitext"]),
            "winogrande": number(row["winogrande"]),
            "arc_easy": number(row["arc_easy"]),
            "arc_challenge": number(row["arc_challenge"]),
            "mmlu": number(row["mmlu"]),
        })

    existing_policies = {row["policy"] for row in rows}
    measured_policies = sorted(path.stem for path in (EXP / "speed").glob("*.json"))
    for policy in measured_policies:
        if policy not in existing_policies:
            rows.append(candidate(policy))

    report = EXP / "report"
    report.mkdir(parents=True, exist_ok=True)
    output = report / "all_results.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ours = [row for row in rows if row["family"] == "ours"]
    uniform = [row for row in rows if row["family"] == "uniform"]
    arc_frontier = frontier(ours, "arc_challenge")
    audits = []
    for reference in uniform:
        dominators = [
            row
            for row in ours
            if row["speedup"] is not None
            and row["arc_challenge"] is not None
            and float(row["speedup"]) >= float(reference["speedup"])
            and float(row["arc_challenge"]) >= float(reference["arc_challenge"])
        ]
        audits.append({
            "uniform": reference["policy"],
            "speedup": reference["speedup"],
            "arc_challenge": reference["arc_challenge"],
            "covered": bool(dominators),
            "example_ours": max(dominators, key=lambda row: float(row["speedup"]))["policy"] if dominators else "",
        })
    audit_path = report / "arc_challenge_coverage.csv"
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)

    metric_audits = []
    for metric_name, (display_name, higher_is_better) in METRICS.items():
        for reference in uniform:
            if reference[metric_name] is None:
                continue
            dominators = [
                row for row in ours
                if row["speedup"] is not None
                and row[metric_name] is not None
                and float(row["speedup"]) >= float(reference["speedup"])
                and (
                    float(row[metric_name]) >= float(reference[metric_name])
                    if higher_is_better
                    else float(row[metric_name]) <= float(reference[metric_name])
                )
            ]
            metric_audits.append({
                "metric": display_name,
                "uniform": reference["policy"],
                "uniform_speedup": reference["speedup"],
                "uniform_quality": reference[metric_name],
                "covered": bool(dominators),
                "example_ours": max(dominators, key=lambda row: float(row["speedup"]))["policy"] if dominators else "",
            })
    metric_audit_path = report / "all_metric_coverage.csv"
    with metric_audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_audits[0]))
        writer.writeheader()
        writer.writerows(metric_audits)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    old = [row for row in ours if row["source"] == "061/060" and row["arc_challenge"] is not None]
    new = [row for row in ours if row["source"] != "061/060" and row["arc_challenge"] is not None]
    ax.scatter([row["speedup"] for row in old], [row["arc_challenge"] for row in old],
               s=42, color="#64748b", alpha=0.55, label="Ours (existing measured)")
    ax.scatter([row["speedup"] for row in new], [row["arc_challenge"] for row in new],
               s=64, color="#1f2937", label="Ours (refined measured)", zorder=3)
    ax.plot([row["speedup"] for row in arc_frontier], [row["arc_challenge"] for row in arc_frontier],
            color="#1f2937", linewidth=2.2, marker="o", label="Ours non-dominated frontier", zorder=2)
    ax.scatter([row["speedup"] for row in uniform], [row["arc_challenge"] for row in uniform],
               marker="s", s=82, color="#dc2626", label="Uniform baselines", zorder=4)
    for row in uniform:
        ax.annotate(str(row["policy"]), (float(row["speedup"]), float(row["arc_challenge"])),
                    xytext=(5, -13), textcoords="offset points", fontsize=8, color="#b91c1c")
    labels = {
        "bridge_dense_nvfp4_120": (5, 7, "bridge-120"),
        "fine_q208": (5, 7, "fine-q208"),
    }
    for row in new:
        if row["policy"] in labels:
            dx, dy, label = labels[str(row["policy"])]
            ax.annotate(label, (float(row["speedup"]), float(row["arc_challenge"])),
                        xytext=(dx, dy), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Measured E2E prefill speedup vs dense BF16")
    ax.set_ylabel("ARC-Challenge accuracy (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(report / "pareto_arc_challenge.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    complete_new = sorted(
        [row for row in new if row["speedup"] is not None],
        key=lambda row: float(row["speedup"]),
    )
    lines = [
        "# Llama3.1 prefill-only uniform-coverage refinement",
        "",
        f"Target: speedup >= {TARGET_SPEEDUP:.4f}x and ARC-Challenge >= {TARGET_ARC_CHALLENGE:.4f}.",
        "",
        "## New measured candidates",
        "",
        "| policy | speedup | delta NLL | WikiText PPL | WinoGrande | ARC-Easy | ARC-Challenge | MMLU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in complete_new:
        values = [row["policy"], row["speedup"], row["delta_nll"], row["wikitext"], row["winogrande"], row["arc_easy"], row["arc_challenge"], row["mmlu"]]
        lines.append("| " + " | ".join("—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value)) for value in values) + " |")
    lines += ["", "## Uniform coverage", "", "| uniform | covered | example ours |", "|---|---:|---|"]
    lines += [f"| {row['uniform']} | {row['covered']} | {row['example_ours'] or '—'} |" for row in audits]
    dense_rows = [row for row in metric_audits if row["uniform"] == "dense_nvfp4"]
    lines += ["", "## Dense-NVFP4 coverage by metric", "", "| metric | covered | example ours |", "|---|---:|---|"]
    lines += [f"| {row['metric']} | {row['covered']} | {row['example_ours'] or '—'} |" for row in dense_rows]
    lines += [
        "",
        "ARC-Challenge is the pre-selected plotting metric for this refinement. The cross-metric audit is reported to expose, rather than hide, metric-specific failures.",
        "",
        "All attempted candidates are retained in `all_results.csv`.",
        "",
        "## Paper figures",
        "",
        "Matched Llama3.1 figures for prefill-only and prefill-decode are under [`paper/`](paper/). The combined figure uses ARC-Challenge for prefill-only and IWSLT SacreBLEU for prefill-decode; CNN/DM ROUGE-L is retained as an appendix alternative.",
        "",
    ]
    (report / "summary.md").write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
