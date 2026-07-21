#!/usr/bin/env python3
"""Package every measured Llama2 Pareto/task point into a paper-facing bundle."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[4]
PREFILL = REPO / "artifacts/debug/037_llama2_prefill_only_pareto"
DECODE = REPO / "artifacts/debug/035_llama2_prefill_decode_e2e_speed_model"
INTERMEDIATE = REPO / "artifacts/debug/036_llama2_prefill_decode_intermediate_points"
OUT = REPO / "artifacts/exports/vllm/ours/llama2-7b-chat/pareto_summary"
BASE = REPO / "artifacts/exports/vllm/baselines/llama2-7b-chat/results/quality"

PRE_RECOMMEND = {
    "ours_point_008": "recommended: high-quality",
    "ours_point_012": "recommended: primary balanced",
    "ours_point_013": "recommended: dense-NVFP4 cover",
    "ours_point_016": "recommended: max-speed endpoint",
}
DEC_RECOMMEND = {
    "point_003": "recommended: high-quality",
    "point_007": "recommended: quality/throughput",
    "i38": "recommended: fast task-validated",
    "point_011": "recommended: max-speed endpoint",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def task_metrics(root: Path) -> dict[str, dict]:
    return {dataset: read_json(root / dataset / "metrics.json") for dataset in ("cnn_dm_1000", "dsum", "IWSLT")}


def core_metric_root(point: str) -> Path:
    continuous = DECODE / "task_quality_continuous/results" / f"point_{point}"
    return continuous if continuous.exists() else DECODE / "task_quality_recovery/results" / f"point_{point}"


def prefill_rows() -> list[dict]:
    rows = []
    for source in read_csv(PREFILL / "arc_challenge/report/arc_challenge_speed_summary.csv"):
        policy = source["label"]
        rows.append({
            "scenario": "prefill-only (B=8, S=2048)", "family": source["family"], "policy": policy,
            "recommendation": PRE_RECOMMEND.get(policy, "baseline" if policy == "dense_bf16" else ""),
            "e2e_ms": float(source["e2e_median_ms"]), "speedup": float(source["speedup_vs_dense"]),
            "speed_source": "measured 5-repeat closure", "arc_norm_pct": 100 * float(source["arc_acc_norm"]),
            "cnn_rougel": None, "cnn_bertscore": None, "dsum_rougel": None, "dsum_bertscore": None,
            "iwslt_rougel": None, "iwslt_bleu": None, "delta_nll": None,
            "task_status": "evaluated on ARC-Challenge (1172)",
        })
    return rows


def metric_map(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        result.setdefault(row["point"], {})[row["dataset"]] = row
    return result


def decode_rows() -> list[dict]:
    core_summary = read_csv(DECODE / "task_quality_all/summary.csv")
    core = metric_map(core_summary)
    core_nll = {row["point"]: row for row in read_csv(DECODE / "report/formal_util085_actual_nll_summary.csv")}
    baseline = {}
    for row in read_csv(DECODE / "task_quality_all/report/all_task_pareto_points.csv"):
        if row["family"] == "baseline":
            baseline.setdefault(row["label"], {})[row["dataset"]] = row
    dense_ms = float(core_nll["0"]["e2e_median_ms"])
    rows: list[dict] = []
    for policy, metrics in baseline.items():
        first = next(iter(metrics.values()))
        speedup = float(first["speedup"])
        raw = task_metrics(BASE / policy)
        rows.append({
            "scenario": "prefill-decode (B=16, S=2048, O=80)", "family": "uniform", "policy": policy,
            "recommendation": "baseline" if policy == "dense_bf16" else "",
            "e2e_ms": dense_ms / speedup, "speedup": speedup, "speed_source": "measured task-run speed (E2E derived)",
            "arc_norm_pct": None, "cnn_rougel": float(metrics["cnn_dm_1000"]["score"]),
            "cnn_bertscore": float(raw["cnn_dm_1000"]["bert_score_percent"]),
            "dsum_rougel": float(metrics["dsum"]["score"]), "dsum_bertscore": float(raw["dsum"]["bert_score_percent"]),
            "iwslt_rougel": float(raw["IWSLT"]["rougeL_percent"]), "iwslt_bleu": float(metrics["IWSLT"]["score"]),
            "delta_nll": None, "task_status": "evaluated on all three tasks",
        })
    for point, metrics in core.items():
        source = core_nll[point]
        policy = f"point_{int(point):03d}"
        raw = task_metrics(core_metric_root(point))
        rows.append({
            "scenario": "prefill-decode (B=16, S=2048, O=80)", "family": "ours", "policy": policy,
            "recommendation": DEC_RECOMMEND.get(policy, "identity / dense reference" if point == "0" else ""),
            "e2e_ms": float(source["e2e_median_ms"]), "speedup": float(source["speedup_vs_point0"]),
            "speed_source": "measured formal closure", "arc_norm_pct": None,
            "cnn_rougel": float(metrics["cnn_dm_1000"]["rougeL_percent"]),
            "cnn_bertscore": float(raw["cnn_dm_1000"]["bert_score_percent"]),
            "dsum_rougel": float(metrics["dsum"]["rougeL_percent"]),
            "dsum_bertscore": float(raw["dsum"]["bert_score_percent"]),
            "iwslt_rougel": float(raw["IWSLT"]["rougeL_percent"]), "iwslt_bleu": float(metrics["IWSLT"]["sacre_bleu"]),
            "delta_nll": float(source["measured_wikitext_delta_nll"]), "task_status": "evaluated on all three tasks",
        })
    intermediate = metric_map(read_csv(INTERMEDIATE / "task_quality_intermediate/summary.csv"))
    intermediate_nll = {row["point"]: row for row in read_csv(INTERMEDIATE / "report/intermediate_actual_nll_summary.csv")}
    for point, metrics in intermediate.items():
        source = intermediate_nll[f"i{point}"]
        policy = f"i{point}"
        raw = task_metrics(INTERMEDIATE / "task_quality_intermediate/full_metrics" / f"point_{point}")
        rows.append({
            "scenario": "prefill-decode (B=16, S=2048, O=80)", "family": "ours-intermediate", "policy": policy,
            "recommendation": DEC_RECOMMEND.get(policy, ""), "e2e_ms": float(source["e2e_median_ms"]),
            "speedup": float(source["speedup_vs_point0"]), "speed_source": "screened-stall measurement*",
            "arc_norm_pct": None, "cnn_rougel": float(raw["cnn_dm_1000"]["rougeL_percent"]),
            "cnn_bertscore": float(raw["cnn_dm_1000"]["bert_score_percent"]),
            "dsum_rougel": float(raw["dsum"]["rougeL_percent"]),
            "dsum_bertscore": float(raw["dsum"]["bert_score_percent"]),
            "iwslt_rougel": float(raw["IWSLT"]["rougeL_percent"]), "iwslt_bleu": float(raw["IWSLT"]["sacre_bleu"]),
            "delta_nll": float(source["measured_wikitext_delta_nll"]), "task_status": "evaluated on all three tasks",
        })
    return rows


def draw_prefill(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    for family, color, marker, label in (("uniform", "#d62728", "s", "Uniform"), ("ours", "#1f2937", "o", "Ours")):
        points = sorted((r for r in rows if r["family"] == family), key=lambda r: r["speedup"])
        ax.plot([r["speedup"] for r in points], [r["arc_norm_pct"] for r in points], color=color,
                linewidth=2.4, marker=marker, markersize=7, label=label)
        for r in points:
            if r["recommendation"] or r["family"] == "uniform":
                ax.annotate(r["policy"].replace("ours_", "ours-"), (r["speedup"], r["arc_norm_pct"]),
                            xytext=(4, 6), textcoords="offset points", fontsize=8, color=color)
    ax.set_title("Llama2-7B-Chat prefill-only: measured speed vs ARC-Challenge")
    ax.set_xlabel("Measured E2E speedup vs dense BF16")
    ax.set_ylabel("ARC-Challenge normalized accuracy (%)")
    ax.grid(alpha=.25); ax.legend(loc="best")
    fig.savefig(OUT / "pareto_prefill_only_arc_challenge.png", dpi=200)
    plt.close(fig)


def draw_decode(rows: list[dict], key: str, metric: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    uniform = [r for r in rows if r["family"] == "uniform" and r[key] is not None]
    core = sorted((r for r in rows if r["family"] == "ours" and r[key] is not None), key=lambda r: r["speedup"])
    intermediate = [r for r in rows if r["family"] == "ours-intermediate" and r[key] is not None]
    ax.scatter([r["speedup"] for r in uniform], [r[key] for r in uniform], c="#d62728", marker="s", s=85, label="Uniform", zorder=3)
    ax.plot([r["speedup"] for r in core], [r[key] for r in core], color="#1f2937", linewidth=2.1,
            marker="o", markersize=6.5, label="Ours: formal Pareto points")
    ax.scatter([r["speedup"] for r in intermediate], [r[key] for r in intermediate], facecolors="none", edgecolors="#2563eb",
               marker="o", s=78, linewidths=1.8, label="Ours: screened intermediate*")
    for r in uniform + core + intermediate:
        if r["recommendation"] or r["family"] == "uniform":
            ax.annotate(r["policy"], (r["speedup"], r[key]), xytext=(4, 6), textcoords="offset points", fontsize=8,
                        color="#1f2937" if r["family"].startswith("ours") else "#b91c1c")
    ax.set_title(f"Llama2-7B-Chat prefill-decode: measured speed vs {metric}")
    ax.set_xlabel("Measured E2E speedup vs dense BF16")
    ax.set_ylabel(metric)
    ax.grid(alpha=.25); ax.legend(loc="best", fontsize=8.5)
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def draw_nll(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    core = sorted((r for r in rows if r["family"] == "ours"), key=lambda r: r["speedup"])
    intermediate = [r for r in rows if r["family"] == "ours-intermediate"]
    ax.plot([r["speedup"] for r in core], [r["delta_nll"] for r in core], color="#1f2937", linewidth=2.1,
            marker="o", markersize=6.5, label="Ours: formal closure")
    ax.scatter([r["speedup"] for r in intermediate], [r["delta_nll"] for r in intermediate], facecolors="none", edgecolors="#2563eb",
               marker="o", s=78, linewidths=1.8, label="Ours: screened intermediate*")
    for r in core + intermediate:
        if r["recommendation"]:
            ax.annotate(r["policy"], (r["speedup"], r["delta_nll"]), xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.set_title("Llama2-7B-Chat prefill-decode: measured speed vs WikiText ΔNLL")
    ax.set_xlabel("Measured E2E speedup vs dense BF16")
    ax.set_ylabel("ΔNLL")
    ax.grid(alpha=.25); ax.legend(loc="best", fontsize=8.5)
    fig.savefig(OUT / "pareto_prefill_decode_wikitext_nll.png", dpi=200)
    plt.close(fig)


def write_table(rows: list[dict]) -> None:
    fields = ["scenario", "family", "policy", "recommendation", "e2e_ms", "speedup", "speed_source", "arc_norm_pct",
              "cnn_rougel", "cnn_bertscore", "dsum_rougel", "dsum_bertscore", "iwslt_rougel", "iwslt_bleu",
              "delta_nll", "task_status"]
    with (OUT / "all_measured_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# Llama2-7B-Chat measured result table", "",
             "Every row contains a measured end-to-end speed and corresponding measured task score. All measured ours points are retained, including each max-speed endpoint; `recommended` is a paper-candidate suggestion rather than a filter.", "",
             "| scenario | family | policy | recommended use | E2E ms | speedup | ARC norm. | CNN R-L | CNN BERTScore | DSum R-L | DSum BERTScore | IWSLT R-L | IWSLT BLEU | ΔNLL | task status | speed source |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append("| {scenario} | {family} | {policy} | {recommendation} | {e2e} | {speedup} | {arc} | {cnn} | {cnn_bert} | {dsum} | {dsum_bert} | {iwslt_rougel} | {iwslt} | {nll} | {task_status} | {source} |".format(
            scenario=r["scenario"], family=r["family"], policy=r["policy"], recommendation=r["recommendation"],
            e2e=number(r["e2e_ms"], 2), speedup=number(r["speedup"]), arc=number(r["arc_norm_pct"]),
            cnn=number(r["cnn_rougel"]), cnn_bert=number(r["cnn_bertscore"]), dsum=number(r["dsum_rougel"]),
            dsum_bert=number(r["dsum_bertscore"]), iwslt_rougel=number(r["iwslt_rougel"]), iwslt=number(r["iwslt_bleu"]),
            nll=number(r["delta_nll"]), task_status=r["task_status"], source=r["speed_source"]))
    lines.extend(["", "## Notes", "",
                  "- Prefill-only evaluates ARC-Challenge normalized accuracy on 1172 examples.",
                  "- Prefill-decode retains both measured metrics per dataset: ROUGE-L/BERTScore for CNN/DM and DialogSum, ROUGE-L/SacreBLEU for IWSLT.",
                  "- The screened intermediate rows were rescored from their original generation JSONL to complete all six task metrics; their speed remains the original stall-screened measurement.",
                  "- `point_011` and `ours_point_016` are the tested max-speed endpoints for their respective scenarios.",
                  "- `screened-stall measurement*` points are included for coverage but have stall-screened timing samples; do not use them for fine-grained timing claims against formal-closure points.",
                  "- Suggested candidates: prefill-only `ours_point_008` (high quality), `ours_point_012` (balanced), `ours_point_013` (dense-NVFP4 coverage), `ours_point_016` (endpoint); prefill-decode `point_003` (high quality), `point_007` (quality/throughput), `i38` (fast), `point_011` (endpoint).", ""])
    (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prefill, decode = prefill_rows(), decode_rows()
    write_table(prefill + decode)
    draw_prefill(prefill)
    draw_decode(decode, "cnn_rougel", "CNN/DM ROUGE-L", "pareto_prefill_decode_cnn_dm.png")
    draw_decode(decode, "cnn_bertscore", "CNN/DM BERTScore (%)", "pareto_prefill_decode_cnn_dm_bertscore.png")
    draw_decode(decode, "dsum_rougel", "DialogSum ROUGE-L", "pareto_prefill_decode_dsum.png")
    draw_decode(decode, "dsum_bertscore", "DialogSum BERTScore (%)", "pareto_prefill_decode_dsum_bertscore.png")
    draw_decode(decode, "iwslt_rougel", "IWSLT ROUGE-L", "pareto_prefill_decode_iwslt_rougel.png")
    draw_decode(decode, "iwslt_bleu", "IWSLT SacreBLEU", "pareto_prefill_decode_iwslt.png")
    draw_nll(decode)


if __name__ == "__main__":
    main()
