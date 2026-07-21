#!/usr/bin/env python3
"""Read-only merger of measured speeds and corrected real-vLLM quality results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/debug/045_runtime_quality_consolidation/report"
PREFILL = {
    "llama2": ROOT / "artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality/results",
    "llama31": ROOT / "artifacts/debug/043_llama31_prefill_only_vllm_runtime_quality/results",
}
NLL = ROOT / "artifacts/debug/044_llama_prefill_decode_vllm_nll/results"
MODELS = {"llama2": "llama2-7b-chat", "llama31": "llama3.1-8b-instruct"}
TASKS = {
    "wikitext": ("wikitext_word_ppl", "word_perplexity,none"),
    "winogrande": ("winogrande_acc_pct", "acc,none"),
    "arc_easy": ("arc_easy_norm_pct", "acc_norm,none"),
    "arc_challenge": ("arc_challenge_norm_pct", "acc_norm,none"),
    "mmlu": ("mmlu_acc_pct", "acc,none"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def speed_rows(model: str, scenario: str) -> list[dict[str, str]]:
    path = ROOT / "artifacts/exports/vllm/ours" / MODELS[model] / "pareto_summary/all_measured_results.csv"
    return [row for row in read_csv(path) if row["scenario"].startswith(scenario)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def prefill_row(model: str, speed: dict[str, str]) -> dict[str, object]:
    policy = speed["policy"]
    root = PREFILL[model] / policy
    row: dict[str, object] = {
        "model": model, "policy": policy, "family": speed["family"],
        "recommendation": speed["recommendation"], "e2e_ms": float(speed["e2e_ms"]),
        "speedup": float(speed["speedup"]), "speed_source": speed["speed_source"],
        "quality_backend": "real vLLM/lm-eval", "quality_status": "complete",
    }
    for task, (column, metric) in TASKS.items():
        data = json.loads((root / task / "full/result.json").read_text())
        value = data["metrics"][metric]
        row[column] = float(value if task == "wikitext" else 100 * value)
    return row


def nll_index(model: str) -> dict[str, dict[str, object]]:
    folder = NLL / f"{model}_prefill_decode"
    result: dict[str, dict[str, object]] = {}
    for path in folder.glob("*.json"):
        if path.name.endswith(".phase_trace.json"):
            continue
        data = json.loads(path.read_text())
        name = path.stem.replace("_runtime", "").replace("_retry", "")
        result[name] = data
    return result


def nll_key(model: str, policy: str) -> str | None:
    if policy == "point_000":
        return "dense_bf16"
    if model == "llama2" and policy == "point_011":
        return "max_speed"
    if model == "llama31" and policy == "point_009_max_speed":
        return "max_speed"
    return policy


def decode_row(model: str, speed: dict[str, str], nll: dict[str, dict[str, object]]) -> dict[str, object]:
    key = nll_key(model, speed["policy"])
    data = nll.get(key) if key else None
    row: dict[str, object] = {
        "model": model, "policy": speed["policy"], "family": speed["family"],
        "recommendation": speed["recommendation"], "e2e_ms": float(speed["e2e_ms"]),
        "speedup": float(speed["speedup"]), "speed_source": speed["speed_source"],
        "avg_nll": "" if data is None else float(data["avg_nll"]),
        "perplexity": "" if data is None else float(data["perplexity"]),
        "quality_backend": "not re-evaluated" if data is None else "real vLLM teacher-forced decode",
        "quality_status": "not re-evaluated" if data is None else "complete",
        "phase_trace": "" if data is None else data["runtime"].get("phase_trace", ""),
        # These are the already-measured vLLM generation-task outputs. They
        # remain valid and are reported beside the newly corrected NLL.
        "cnn_rougel": speed["cnn_rougel"], "cnn_bertscore": speed["cnn_bertscore"],
        "dsum_rougel": speed["dsum_rougel"], "dsum_bertscore": speed["dsum_bertscore"],
        "iwslt_rougel": speed["iwslt_rougel"], "iwslt_bleu": speed["iwslt_bleu"],
        "downstream_source": "existing measured vLLM generation tasks" if speed["cnn_rougel"] else "not available",
    }
    return row


def plot_metric(model: str, scenario: str, rows: list[dict[str, object]], metric: str,
                ylabel: str, filename: str) -> None:
    usable = [row for row in rows if row.get(metric, "") != ""]
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = (("uniform", "s", "#d62728", "uniform"),
              ("ours", "o", "#1f2937", "ours (formal)"),
              ("ours-intermediate", "^", "#8a8a8a", "ours (intermediate)"))
    for family, marker, color, label in styles:
        subset = [r for r in usable if r["family"] == family]
        if not subset:
            continue
        subset.sort(key=lambda row: float(row["speedup"]))
        ax.scatter([r["speedup"] for r in subset], [float(r[metric]) for r in subset],
                   marker=marker, s=58, color=color, label=label, zorder=3)
        if family == "ours" and len(subset) > 1:
            ax.plot([r["speedup"] for r in subset], [float(r[metric]) for r in subset],
                    color=color, linewidth=1.5, alpha=.85, zorder=2)
    ax.set(xlabel=f"Measured {scenario} speedup vs dense BF16", ylabel=ylabel,
           title=f"{model}: {scenario} speed vs {ylabel}")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "pareto" / filename, dpi=220); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pareto").mkdir(exist_ok=True)
    all_prefill, all_decode = [], []
    for model in MODELS:
        prefill = [prefill_row(model, row) for row in speed_rows(model, "prefill-only")]
        decode = [decode_row(model, row, nll_index(model)) for row in speed_rows(model, "prefill-decode")]
        all_prefill.extend(prefill); all_decode.extend(decode)
        for metric, ylabel, name in (
            ("wikitext_word_ppl", "WikiText word PPL (↓)", "wikitext_ppl"),
            ("winogrande_acc_pct", "WinoGrande accuracy (%)", "winogrande"),
            ("arc_easy_norm_pct", "ARC-Easy normalized accuracy (%)", "arc_easy"),
            ("arc_challenge_norm_pct", "ARC-Challenge normalized accuracy (%)", "arc_challenge"),
            ("mmlu_acc_pct", "MMLU accuracy (%)", "mmlu"),
        ):
            plot_metric(model, "prefill-only", prefill, metric, ylabel, f"{model}_prefill_only_{name}.png")
        for metric, ylabel, name in (
            ("avg_nll", "Corrected real-vLLM avg NLL (↓)", "nll"),
            ("cnn_rougel", "CNN/DM ROUGE-L", "cnn_rougel"),
            ("cnn_bertscore", "CNN/DM BERTScore", "cnn_bertscore"),
            ("dsum_rougel", "DialogSum ROUGE-L", "dsum_rougel"),
            ("dsum_bertscore", "DialogSum BERTScore", "dsum_bertscore"),
            ("iwslt_rougel", "IWSLT ROUGE-L", "iwslt_rougel"),
            ("iwslt_bleu", "IWSLT BLEU", "iwslt_bleu"),
        ):
            plot_metric(model, "prefill-decoding", decode, metric, ylabel, f"{model}_prefill_decode_{name}.png")
    write_csv(OUT / "prefill_only_corrected_runtime_quality.csv", all_prefill)
    write_csv(OUT / "prefill_decode_corrected_runtime_nll.csv", all_decode)
    write_csv(OUT / "prefill_decode_downstream_tasks.csv", all_decode)
    lines = ["# Debug 045: corrected runtime-quality consolidation", "",
             "Speed is copied from prior measured vLLM speed closure. Prefill-only quality is from debug 042/043 real vLLM lm-eval; prefill-decoding NLL is from debug 044 real vLLM teacher-forced decoding.",
             "No file under `artifacts/exports/` is modified. `not re-evaluated` rows are historical intermediate speed-only points and are excluded from the corrected NLL plots.", "",
             "## Generated artifacts", "",
             "- `prefill_only_corrected_runtime_quality.csv` — all measured prefill-only speed rows with five corrected task metrics.",
             "- `prefill_decode_corrected_runtime_nll.csv` — all existing prefill-decode speed rows with corrected NLL where applicable.",
             "- `prefill_decode_downstream_tasks.csv` — the same prefill-decode rows with all three historical real-vLLM generation-task metric pairs.",
             "- `pareto/` — one speed-quality plot per prefill-only or prefill-decoding metric, with uniform, ours-formal and intermediate styles.", "",
             "## Prefill-only: measured speed and corrected runtime quality", "",
             "| model | policy | family | speedup | WikiText PPL ↓ | WinoGrande (%) | ARC-Easy (%) | ARC-Challenge (%) | MMLU (%) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in all_prefill:
        lines.append(
            f"| {row['model']} | {row['policy']} | {row['family']} | {row['speedup']:.3f} | "
            f"{row['wikitext_word_ppl']:.3f} | {row['winogrande_acc_pct']:.2f} | {row['arc_easy_norm_pct']:.2f} | "
            f"{row['arc_challenge_norm_pct']:.2f} | {row['mmlu_acc_pct']:.2f} |"
        )
    lines.extend(["", "## Prefill-decoding: measured speed and corrected real-vLLM NLL", "",
                  "| model | policy | family | speedup | avg NLL ↓ | PPL ↓ | status |",
                  "|---|---|---|---:|---:|---:|---|"])
    for row in all_decode:
        nll = "—" if row["quality_status"] != "complete" else f"{row['avg_nll']:.6f}"
        ppl = "—" if row["quality_status"] != "complete" else f"{row['perplexity']:.4f}"
        lines.append(f"| {row['model']} | {row['policy']} | {row['family']} | {row['speedup']:.3f} | {nll} | {ppl} | {row['quality_status']} |")
    lines.extend(["", "## Prefill-decoding: existing measured downstream generation tasks", "",
                  "CNN/DM and DialogSum use ROUGE-L / BERTScore; IWSLT uses ROUGE-L / BLEU. These generation-task values are inherited unchanged from the prior measured vLLM task runs; they are not proxy-quality values.", "",
                  "| model | policy | speedup | CNN R-L | CNN BERTScore | DSum R-L | DSum BERTScore | IWSLT R-L | IWSLT BLEU |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in all_decode:
        metric = lambda key: "—" if row[key] == "" else f"{float(row[key]):.3f}"
        lines.append(f"| {row['model']} | {row['policy']} | {row['speedup']:.3f} | {metric('cnn_rougel')} | {metric('cnn_bertscore')} | {metric('dsum_rougel')} | {metric('dsum_bertscore')} | {metric('iwslt_rougel')} | {metric('iwslt_bleu')} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
