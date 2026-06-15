#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from common_pareto import DEBUG_ROOT, read_csv, write_csv, write_json


BASELINE_METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize WikiText-2 PPL Pareto validation results.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root
    summary_dir = args.summary_dir or root / "summary" / "llama2_wikitext2_ppl_pareto"
    summary_dir.mkdir(parents=True, exist_ok=True)

    pareto_rows = load_pareto_ppl(root)
    stable_rows = load_stable_e2e(root)
    baseline_rows = load_baseline_ppl(root)
    joined = join_pareto(pareto_rows, stable_rows)
    baselines = enrich_baselines(baseline_rows)

    write_csv(root / "validation" / "wikitext2_ppl" / "pareto_wikitext2_test_ppl.csv", pareto_rows)
    write_csv(summary_dir / "pareto_wikitext2_ppl_summary.csv", joined)
    write_csv(summary_dir / "uniform_wikitext2_ppl_summary.csv", baselines)
    write_json(summary_dir / "metadata.json", {"points": len(joined), "baselines": len(baselines)})

    plot_predicted(joined, baselines, summary_dir / "pareto_predicted_speed_vs_wikitext2_ppl.png")
    plot_measured(joined, baselines, summary_dir / "pareto_measured_speed_vs_wikitext2_ppl.png")
    write_report(summary_dir / "README.md", joined, baselines)
    print(f"wrote {summary_dir}")


def load_pareto_ppl(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "validation" / "wikitext2_ppl").glob("pareto_wikitext2_test_ppl_shard_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path))
    dedup: dict[int, dict[str, Any]] = {}
    for row in rows:
        dedup[int(row["point_index"])] = row
    return [dedup[idx] for idx in sorted(dedup)]


def load_stable_e2e(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "validation" / "stable_e2e_repeats" / "stable_e2e_repeats_all_points.csv"
    if not path.exists():
        return {}
    return {int(row["point_index"]): row for row in read_csv(path)}


def load_baseline_ppl(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in BASELINE_METHODS:
        path = root / "validation" / "wikitext2_ppl" / f"baseline_{method}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            method_rows = list(reader)
        if not method_rows:
            continue
        row = dict(method_rows[0])
        row["method"] = method
        rows.append(row)
    return rows


def join_pareto(ppl_rows: list[dict[str, Any]], stable_rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    dense_pred = float(ppl_rows[0]["predicted_latency_ms"])
    dense_ppl = float(ppl_rows[0]["ppl"])
    dense_nll = float(ppl_rows[0]["nll"])
    out: list[dict[str, Any]] = []
    for row in ppl_rows:
        idx = int(row["point_index"])
        stable = stable_rows.get(idx, {})
        predicted_latency = float(row["predicted_latency_ms"])
        ppl = float(row["ppl"])
        nll = float(row["nll"])
        item: dict[str, Any] = {
            "point_index": idx,
            "quality_cost": float(row["quality_cost"]),
            "replaced_modules": int(row["replaced_modules"]),
            "predicted_latency_ms": predicted_latency,
            "predicted_speedup_vs_p0": dense_pred / predicted_latency,
            "wikitext2_nll": nll,
            "wikitext2_ppl": ppl,
            "nll_delta_vs_p0": nll - dense_nll,
            "ppl_delta_vs_p0": ppl - dense_ppl,
            "ppl_rel_delta_pct_vs_p0": (ppl / dense_ppl - 1.0) * 100.0,
            "tokens": int(row["tokens"]),
            "blocks": int(row["blocks"]),
        }
        if stable:
            item.update(
                {
                    "measured_e2e_ms": float(stable["e2e_total_mean_ms"]),
                    "measured_speedup_vs_p0": float(stable["e2e_speedup_vs_point0"]),
                    "backend_counts": stable.get("backend_counts", ""),
                }
            )
        else:
            item.update({"measured_e2e_ms": "", "measured_speedup_vs_p0": "", "backend_counts": ""})
        out.append(item)
    return out


def enrich_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense = next(row for row in rows if row["method"] == "dense_bf16")
    dense_pred = float(dense["predicted_latency_ms"])
    dense_ppl = float(dense["ppl"])
    dense_nll = float(dense["nll"])
    out: list[dict[str, Any]] = []
    for row in rows:
        predicted_latency = float(row["predicted_latency_ms"])
        ppl = float(row["ppl"])
        nll = float(row["nll"])
        out.append(
            {
                "method": row["method"],
                "quality_cost": float(row["quality_cost"]),
                "replaced_modules": int(row["replaced_modules"]),
                "predicted_latency_ms": predicted_latency,
                "predicted_speedup_vs_dense_bf16": dense_pred / predicted_latency,
                "wikitext2_nll": nll,
                "wikitext2_ppl": ppl,
                "nll_delta_vs_dense_bf16": nll - dense_nll,
                "ppl_delta_vs_dense_bf16": ppl - dense_ppl,
                "ppl_rel_delta_pct_vs_dense_bf16": (ppl / dense_ppl - 1.0) * 100.0,
                "tokens": int(row["tokens"]),
                "blocks": int(row["blocks"]),
            }
        )
    return out


def plot_predicted(pareto: list[dict[str, Any]], baselines: list[dict[str, Any]], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        [float(row["predicted_speedup_vs_p0"]) for row in pareto],
        [float(row["wikitext2_ppl"]) for row in pareto],
        marker="o",
        label="Pareto policy points",
    )
    for row in baselines:
        plt.scatter(float(row["predicted_speedup_vs_dense_bf16"]), float(row["wikitext2_ppl"]), marker="x", s=70)
        plt.annotate(row["method"], (float(row["predicted_speedup_vs_dense_bf16"]), float(row["wikitext2_ppl"])))
    plt.xlabel("Predicted speedup vs dense_bf16")
    plt.ylabel("WikiText-2 test PPL")
    plt.title("Llama2-7B normal02: predicted speed vs WikiText-2 PPL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_measured(pareto: list[dict[str, Any]], baselines: list[dict[str, Any]], path: Path) -> None:
    measured = [row for row in pareto if row["measured_speedup_vs_p0"] != ""]
    plt.figure(figsize=(8, 5))
    plt.plot(
        [float(row["measured_speedup_vs_p0"]) for row in measured],
        [float(row["wikitext2_ppl"]) for row in measured],
        marker="o",
        label="Pareto points with stable E2E timing",
    )
    for row in baselines:
        plt.scatter(float(row["predicted_speedup_vs_dense_bf16"]), float(row["wikitext2_ppl"]), marker="x", s=70)
        plt.annotate(row["method"], (float(row["predicted_speedup_vs_dense_bf16"]), float(row["wikitext2_ppl"])))
    plt.xlabel("Speedup vs dense_bf16; Pareto uses measured E2E, baselines use predicted cost")
    plt.ylabel("WikiText-2 test PPL")
    plt.title("Llama2-7B normal02: measured speed subset vs WikiText-2 PPL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_report(path: Path, pareto: list[dict[str, Any]], baselines: list[dict[str, Any]]) -> None:
    best_fast = max((row for row in pareto if row["measured_speedup_vs_p0"] != ""), key=lambda row: float(row["measured_speedup_vs_p0"]))
    p0 = pareto[0]
    lines = [
        "# Llama2-7B WikiText-2 PPL Pareto Summary",
        "",
        "This report replaces ARC-Challenge as the current final quality check for the normal02 prefill+decode scenario. Quality is measured by teacher-forced WikiText-2 test perplexity; speed remains the existing normal02 prefill+decode latency model and stable E2E timing where available.",
        "",
        "## Main Observations",
        "",
        f"- Dense bf16 P0: PPL {float(p0['wikitext2_ppl']):.4f}, NLL {float(p0['wikitext2_nll']):.6f}.",
        f"- Fastest measured Pareto point P{int(best_fast['point_index'])}: measured speedup {float(best_fast['measured_speedup_vs_p0']):.4f}x, PPL {float(best_fast['wikitext2_ppl']):.4f}, relative PPL delta {float(best_fast['ppl_rel_delta_pct_vs_p0']):.2f}%.",
        "- The Pareto policies increase PPL smoothly as predicted speed improves; P9 matches the all-nvfp4 quality endpoint but is faster than uniform marlin/dense-nvfp4 in the normal02 cost model.",
        "- dense_nvfp4, marlin_nvfp4, and dense_nvfp4_prefill_marlin_decode have identical PPL here because all three use the same dense nvfp4 compressed weights for quality; their difference is only the runtime backend assignment.",
        "- P1-P3 now have PPL but still do not have stable measured E2E repeats; plots include a predicted-speed curve for all points and a measured-speed curve for the already measured subset.",
        "",
        "## Pareto Points",
        "",
        "| point | pred speedup | measured speedup | PPL | PPL delta % | replaced modules |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pareto:
        measured = row["measured_speedup_vs_p0"]
        measured_text = "" if measured == "" else f"{float(measured):.4f}x"
        lines.append(
            f"| P{int(row['point_index'])} | {float(row['predicted_speedup_vs_p0']):.4f}x | {measured_text} | "
            f"{float(row['wikitext2_ppl']):.4f} | {float(row['ppl_rel_delta_pct_vs_p0']):.2f}% | {int(row['replaced_modules'])} |"
        )
    lines.extend(
        [
            "",
            "## Uniform Baselines",
            "",
            "| method | predicted speedup | PPL | PPL delta % | replaced modules |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in baselines:
        lines.append(
            f"| {row['method']} | {float(row['predicted_speedup_vs_dense_bf16']):.4f}x | "
            f"{float(row['wikitext2_ppl']):.4f} | {float(row['ppl_rel_delta_pct_vs_dense_bf16']):.2f}% | {int(row['replaced_modules'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `pareto_wikitext2_ppl_summary.csv`: joined Pareto PPL plus stable E2E timing where available.",
            "- `uniform_wikitext2_ppl_summary.csv`: uniform method PPL baselines.",
            "- `pareto_predicted_speed_vs_wikitext2_ppl.png`: all Pareto points and uniform baselines on predicted speed.",
            "- `pareto_measured_speed_vs_wikitext2_ppl.png`: measured E2E subset, with uniform baselines shown using predicted speed for reference.",
            "",
            "## Next Steps",
            "",
            "1. Run stable E2E repeats for P1-P3, so the measured-speed curve covers every PPL-validated point.",
            "2. Add measured E2E timings for uniform marlin_nvfp4 and dense_nvfp4_prefill_marlin_decode, because they are the important speed baselines under the same final PPL.",
            "3. Use WikiText-2 PPL as the primary quality axis for the next optimization audit; ARC can remain optional downstream task context, not the final metric for this prefill+decode scenario.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
