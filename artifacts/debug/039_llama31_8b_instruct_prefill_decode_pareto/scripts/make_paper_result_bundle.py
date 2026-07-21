#!/usr/bin/env python3
"""Package measured Llama-3.1 results into one table and four Pareto figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[4]
PREFILL = REPO / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto"
DECODE = Path(__file__).resolve().parents[1]
BASE = REPO / "artifacts/exports/vllm/baselines/llama3.1-8b-instruct/results"
MAX_SPEED = REPO / "artifacts/exports/vllm/ours/llama3.1-8b-instruct/max_speed/prefill_decode/results/quality"
OUT = REPO / "artifacts/exports/vllm/ours/llama3.1-8b-instruct/pareto_summary"

UNIFORM = ("dense_bf16", "dense_nvfp4", "marlin_nvfp4", "sparse_bf16", "sparse_nvfp4")
PRE_RECOMMEND = {
    "ours_point_6": "recommended: near-lossless",
    "ours_point_8": "recommended: dense-NVFP4-cover",
    "ours_point_11": "optional: high-speed trade-off",
}
DEC_RECOMMEND = {
    "point_000": "identity / dense reference",
    "point_002": "recommended: high-quality",
    "point_004": "recommended: primary balanced",
    "point_006": "recommended: fast task-validated",
    "point_008": "recommended: high-speed task-validated",
    "point_009_max_speed": "recommended: max-speed endpoint",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def prefill_rows() -> list[dict]:
    rows = []
    for source in csv_rows(PREFILL / "arc_challenge/report/arc_challenge_speed_summary.csv"):
        policy = source["label"]
        rows.append({
            "scenario": "prefill-only (B=8, S=2048)",
            "family": source["family"], "policy": policy,
            "recommendation": PRE_RECOMMEND.get(policy, "baseline" if policy == "dense_bf16" else ""),
            "e2e_ms": float(source["e2e_median_ms"]), "speedup": float(source["speedup_vs_dense"]),
            "speed_source": "fresh 5-repeat closure",
            "arc_norm_pct": 100 * float(source["arc_acc_norm"]),
            "cnn_rougel": None, "cnn_bertscore": None, "dsum_rougel": None, "dsum_bertscore": None,
            "iwslt_rougel": None, "iwslt_bleu": None, "delta_nll": None,
        })
    return rows


def decode_rows() -> list[dict]:
    closure = {r["policy_id"]: r for r in csv_rows(DECODE / "closure/summary.csv")}
    legacy_speed = {r["method"]: float(r["e2e_median_ms"])
                    for r in csv_rows(BASE / "summary/speed_summary.csv")
                    if r["scenario"] == "prefill_decode"}
    dense_legacy = legacy_speed["dense_bf16"]
    nll = {r["policy_id"]: float(r["target_delta_nll"])
           for r in csv_rows(DECODE / "nll/prefill_decode.csv")}
    nll_policy = {"dense_bf16": "p00", "dense_nvfp4": "p01", "sparse_bf16": "p02",
                  "sparse_nvfp4": "p03", "marlin_nvfp4": "p04"}
    rows: list[dict] = []
    for method in UNIFORM:
        if method in {"dense_bf16", "dense_nvfp4", "sparse_bf16"}:
            key = "point_000" if method == "dense_bf16" else method
            source = closure[key]
            e2e = float(closure["point_000"]["speed_median_ms"]) if method == "dense_bf16" else float(source["speed_median_ms"])
            speedup = 1.0 if method == "dense_bf16" else float(source["speedup_vs_dense"])
            speed_source = "fresh continuous closure"
        else:
            e2e = legacy_speed[method]
            speedup = dense_legacy / e2e
            speed_source = "frozen legacy runner*"
        metrics = {dataset: load_json(BASE / "quality" / method / dataset / "metrics.json")
                   for dataset in ("cnn_dm_1000", "dsum", "IWSLT")}
        rows.append({
            "scenario": "prefill-decode (B=16, S=2048, O=80)", "family": "uniform", "policy": method,
            "recommendation": "baseline" if method == "dense_bf16" else "",
            "e2e_ms": e2e, "speedup": speedup, "speed_source": speed_source,
            "arc_norm_pct": None, "cnn_rougel": float(metrics["cnn_dm_1000"]["rougeL_percent"]),
            "cnn_bertscore": float(metrics["cnn_dm_1000"]["bert_score_percent"]),
            "dsum_rougel": float(metrics["dsum"]["rougeL_percent"]),
            "dsum_bertscore": float(metrics["dsum"]["bert_score_percent"]),
            "iwslt_rougel": float(metrics["IWSLT"]["rougeL_percent"]),
            "iwslt_bleu": float(metrics["IWSLT"]["sacre_bleu"]),
            "delta_nll": nll[nll_policy[method]],
        })
    for point in ("point_000", "point_002", "point_004", "point_006", "point_008", "point_009_max_speed"):
        key = point.replace("_max_speed", "")
        if point == "point_000":
            paths, task_status = BASE / "quality" / "dense_bf16", "same policy as dense BF16"
        elif point in {"point_002", "point_004", "point_006", "point_008"}:
            paths, task_status = DECODE / "closure/tasks" / point / "results/quality", "evaluated on all three tasks"
        elif point.endswith("max_speed"):
            paths, task_status = MAX_SPEED, "pre-existing max-speed task run"
        else:
            paths, task_status = None, "closure only; downstream not evaluated"
        metrics = ({dataset: load_json(paths / dataset / "metrics.json")
                    for dataset in ("cnn_dm_1000", "dsum", "IWSLT")}
                   if paths is not None else {})
        source = closure[key]
        rows.append({
            "scenario": "prefill-decode (B=16, S=2048, O=80)", "family": "ours", "policy": point,
            "recommendation": DEC_RECOMMEND[point], "e2e_ms": float(source["speed_median_ms"]),
            "speedup": float(source["speedup_vs_dense"]), "speed_source": "fresh continuous closure",
            "arc_norm_pct": None,
            "cnn_rougel": float(metrics["cnn_dm_1000"]["rougeL_percent"]) if metrics else None,
            "cnn_bertscore": float(metrics["cnn_dm_1000"]["bert_score_percent"]) if metrics else None,
            "dsum_rougel": float(metrics["dsum"]["rougeL_percent"]) if metrics else None,
            "dsum_bertscore": float(metrics["dsum"]["bert_score_percent"]) if metrics else None,
            "iwslt_rougel": float(metrics["IWSLT"]["rougeL_percent"]) if metrics else None,
            "iwslt_bleu": float(metrics["IWSLT"]["sacre_bleu"]) if metrics else None,
            "delta_nll": float(source["actual_delta_nll"]), "task_status": task_status,
        })
    return rows


def draw_prefill(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    for family, color, marker, label in (("uniform", "#d62728", "s", "Uniform"), ("ours", "#1f2937", "o", "Ours")):
        points = sorted((r for r in rows if r["family"] == family), key=lambda r: r["speedup"])
        ax.plot([r["speedup"] for r in points], [r["arc_norm_pct"] for r in points], color=color,
                linewidth=2.7, marker=marker, markersize=8, label=label)
        for r in points:
            tag = r["policy"].replace("ours_", "ours-")
            ax.annotate(tag, (r["speedup"], r["arc_norm_pct"]), xytext=(5, 6), textcoords="offset points",
                        fontsize=8.5, color=color)
    ax.set_title("Llama-3.1-8B-Instruct prefill-only: speed vs ARC-Challenge")
    ax.set_xlabel("Measured E2E speedup vs dense BF16")
    ax.set_ylabel("ARC-Challenge normalized accuracy (%)")
    ax.grid(alpha=.25); ax.legend(loc="best")
    fig.savefig(OUT / "pareto_prefill_only_arc_challenge.png", dpi=200)
    plt.close(fig)


def draw_decode(rows: list[dict], dataset: str, metric: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    uniform = [r for r in rows if r["family"] == "uniform"]
    ours = sorted((r for r in rows if r["family"] == "ours" and r[dataset] is not None), key=lambda r: r["speedup"])
    for source, marker, label in (("fresh continuous closure", "s", "Uniform (continuous)"),
                                  ("frozen legacy runner*", "X", "Uniform (legacy*)")):
        points = [r for r in uniform if r["speed_source"] == source]
        if points:
            ax.scatter([r["speedup"] for r in points], [r[dataset] for r in points], c="#d62728", marker=marker,
                       s=100, label=label, zorder=3)
            for r in points:
                ax.annotate(r["policy"], (r["speedup"], r[dataset]), xytext=(5, 6), textcoords="offset points",
                            fontsize=8.5, color="#d62728")
    ax.plot([r["speedup"] for r in ours], [r[dataset] for r in ours], color="#1f2937", linewidth=2.8,
            marker="o", markersize=9, label="Ours: Pareto policies")
    labels = {"point_000": "ours-identity", "point_002": "ours-high-Q", "point_004": "ours-mid", "point_006": "ours-fast", "point_008": "ours-faster", "point_009_max_speed": "ours-max"}
    for r in ours:
        ax.annotate(labels[r["policy"]], (r["speedup"], r[dataset]), xytext=(5, 6), textcoords="offset points",
                    fontsize=9, color="#1f2937")
    all_x = [r["speedup"] for r in rows]
    ax.set_xlim(min(all_x) - .06, max(all_x) + .16)
    ax.set_title(f"Llama-3.1-8B-Instruct prefill-decode: speed vs {metric}")
    ax.set_xlabel("Measured E2E speedup vs dense BF16")
    ax.set_ylabel(metric)
    ax.grid(alpha=.25); ax.legend(loc="best", fontsize=9)
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def draw_nll(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    uniform = [r for r in rows if r["family"] == "uniform"]
    ours = sorted((r for r in rows if r["family"] == "ours"), key=lambda r: r["speedup"])
    for source, marker, label in (("fresh continuous closure", "s", "Uniform (continuous)"),
                                  ("frozen legacy runner*", "X", "Uniform (legacy*)")):
        points = [r for r in uniform if r["speed_source"] == source]
        visible = [r for r in points if r["delta_nll"] <= 60]
        offscale = [r for r in points if r["delta_nll"] > 60]
        ax.scatter([r["speedup"] for r in visible], [r["delta_nll"] for r in visible], c="#d62728", marker=marker,
                   s=100, label=label, zorder=3)
        for r in offscale:
            ax.scatter([r["speedup"]], [59], c="#d62728", marker="^", s=120, zorder=4)
            ax.annotate(f"{r['policy']} ΔNLL={r['delta_nll']:.1f}", (r["speedup"], 59), xytext=(5, 5),
                        textcoords="offset points", fontsize=8.5, color="#d62728")
    ax.plot([r["speedup"] for r in ours], [r["delta_nll"] for r in ours], color="#1f2937", linewidth=2.8,
            marker="o", markersize=8.5, label="Ours: measured closure")
    for r in ours:
        ax.annotate(r["policy"].replace("point_", "p"), (r["speedup"], r["delta_nll"]), xytext=(5, 6),
                    textcoords="offset points", fontsize=8.5, color="#1f2937")
    ax.set_ylim(-3, 62)
    ax.set_title("Llama-3.1-8B-Instruct prefill-decode: speed vs measured WikiText ΔNLL")
    ax.set_xlabel("Measured E2E speedup vs dense BF16")
    ax.set_ylabel("ΔNLL (100 WikiText blocks)")
    ax.grid(alpha=.25); ax.legend(loc="upper right", fontsize=9)
    fig.savefig(OUT / "pareto_prefill_decode_wikitext_nll.png", dpi=200)
    plt.close(fig)


def write_table(rows: list[dict]) -> None:
    fields = ["scenario", "family", "policy", "recommendation", "e2e_ms", "speedup", "speed_source",
              "arc_norm_pct", "cnn_rougel", "cnn_bertscore", "dsum_rougel", "dsum_bertscore", "iwslt_rougel",
              "iwslt_bleu", "delta_nll", "task_status"]
    with (OUT / "all_measured_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Llama-3.1-8B-Instruct measured result table", "",
             "All listed rows have a measured speed. All six prefill-decode closure points have measured WikiText NLL; five selected points have full downstream task scores. `recommended` labels are suggestions, not filtering.",
             "", "| scenario | family | policy | recommended use | E2E ms | speedup | ARC norm. | CNN R-L | CNN BERTScore | DSum R-L | DSum BERTScore | IWSLT R-L | IWSLT BLEU | ΔNLL | task status | speed source |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append("| {scenario} | {family} | {policy} | {recommendation} | {e2e} | {speedup} | {arc} | {cnn} | {cnn_bert} | {dsum} | {dsum_bert} | {iwslt_rougel} | {iwslt} | {nll} | {task_status} | {source} |".format(
            scenario=r["scenario"], family=r["family"], policy=r["policy"], recommendation=r["recommendation"] or "",
            e2e=number(r["e2e_ms"], 2), speedup=number(r["speedup"]), arc=number(r["arc_norm_pct"]),
            cnn=number(r["cnn_rougel"]), cnn_bert=number(r["cnn_bertscore"]), dsum=number(r["dsum_rougel"]),
            dsum_bert=number(r["dsum_bertscore"]), iwslt_rougel=number(r["iwslt_rougel"]), iwslt=number(r["iwslt_bleu"]),
            nll=number(r["delta_nll"]), task_status=r.get("task_status", "evaluated"), source=r["speed_source"]))
    lines.extend(["", "## Notes", "",
                  "- Prefill-only uses ARC-Challenge normalized accuracy over 1172 examples.",
                  "- Prefill-decode retains both measured metrics per dataset: ROUGE-L/BERTScore for CNN/DM and DialogSum, ROUGE-L/SacreBLEU for IWSLT.",
                  "- `fresh continuous closure` denotes the 6-warmup / 5-measurement phase-continuous protocol.  "
                  "`frozen legacy runner*` is retained because no continuous remeasurement exists for those two uniform methods; "
                  "they are visibly distinguished in the task figures and should not be used for a fine-grained speed claim.",
                  "- Recommended paper candidates: prefill-only `ours_point_6` (near-lossless) and `ours_point_8` (dense-NVFP4 coverage); "
                  "prefill-decode `point_002` (high quality), `point_004` (balanced primary), `point_006`/`point_008` (task-validated fast points), and `point_009_max_speed` (endpoint).", ""])
    (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prefill = prefill_rows()
    decode = decode_rows()
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
