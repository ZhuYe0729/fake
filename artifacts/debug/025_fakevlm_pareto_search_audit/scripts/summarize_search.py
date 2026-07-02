#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from common_search_audit import DEBUG_ROOT, DEFAULT_BATCH_SIZE, SOURCE_024_ROOT, f, read_csv, report_024_rows, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FakeVLM search-audit results.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_validation_rows(args.output_root)
    write_csv(args.output_root / "summary" / "search_results.csv", rows)
    searched = [row for row in rows if row.get("family") != "reference_024"]
    measured_reference = measured_024_reference_rows(rows)
    frontier = nondominated(searched)
    write_csv(args.output_root / "summary" / "non_dominated_search_frontier.csv", frontier)
    reference = measured_reference or report_024_rows(args.batch_size)
    if measured_reference:
        write_csv(args.output_root / "summary" / "measured_024_reference.csv", measured_reference)
    gap = gap_to_reference(reference, searched)
    write_csv(args.output_root / "summary" / "gap_to_024_pareto.csv", gap)
    write_report(args.output_root, rows, searched, frontier, reference, gap, bool(measured_reference))
    plot(args.output_root, searched, frontier, reference)
    plot_speedup(args.output_root, searched, frontier, reference)
    plot_speedup_with_uniform(args.output_root, searched, frontier, reference, uniform_024_rows(args.batch_size))
    write_json(
        args.output_root / "summary" / "summary_metadata.json",
        {
            "rows": len(rows),
            "searched_rows": len(searched),
            "frontier_rows": len(frontier),
            "reference_rows": len(reference),
            "reference_source": "measured_025" if measured_reference else "024_report",
        },
    )
    print(f"wrote summary rows={len(rows)} frontier={len(frontier)}")


def read_validation_rows(output_root: Path) -> list[dict[str, Any]]:
    policy_rows = {row["key"]: row for row in read_csv(output_root / "search" / "search_policies.csv")}
    rows = []
    for path in sorted((output_root / "validation" / "policies").glob("*.csv")):
        for row in read_csv(path):
            policy_row = policy_rows.get(row.get("key", ""))
            if policy_row is not None:
                row = {**row, **policy_row, **actual_fields(row)}
            rows.append(row)
    return rows


def actual_fields(row: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("actual_", "e2e_", "samples_", "global_", "total_", "subset_", "accuracy_", "calib_", "replaced_", "skipped_", "runtime_", "warmup", "iters", "input_tokens")
    return {key: value for key, value in row.items() if key.startswith(prefixes) or key in {"key"}}


def nondominated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [row for row in rows if row.get("global_accuracy") and row.get("e2e_prefill_mean_ms")]
    out = []
    for row in complete:
        latency = f(row, "e2e_prefill_mean_ms")
        acc = f(row, "global_accuracy")
        dominated = False
        for other in complete:
            if other is row:
                continue
            other_latency = f(other, "e2e_prefill_mean_ms")
            other_acc = f(other, "global_accuracy")
            if other_latency <= latency and other_acc >= acc and (other_latency < latency or other_acc > acc):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda row: (f(row, "e2e_prefill_mean_ms"), -f(row, "global_accuracy")))


def measured_024_reference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("family") != "reference_024":
            continue
        out.append(
            {
                **row,
                "label": row.get("policy_name", row.get("key", "")),
                "fakeclue_accuracy": row.get("global_accuracy", ""),
            }
        )
    return sorted(out, key=lambda row: f(row, "parent_point"))


def gap_to_reference(reference: list[dict[str, Any]], searched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    complete = [row for row in searched if row.get("global_accuracy") and row.get("e2e_prefill_mean_ms")]
    for ref in reference:
        ref_acc = f(ref, "fakeclue_accuracy")
        ref_latency = f(ref, "e2e_prefill_mean_ms")
        candidates = [row for row in complete if f(row, "global_accuracy") >= ref_acc and f(row, "e2e_prefill_mean_ms") < ref_latency]
        best = min(candidates, key=lambda row: f(row, "e2e_prefill_mean_ms"), default=None)
        if best is None:
            out.append(
                {
                    "reference_label": ref.get("label", ""),
                    "reference_accuracy": ref_acc,
                    "reference_e2e_prefill_mean_ms": ref_latency,
                    "dominating_search_key": "",
                    "search_accuracy": "",
                    "search_e2e_prefill_mean_ms": "",
                    "latency_improvement_ms": 0.0,
                    "latency_improvement_pct": 0.0,
                }
            )
            continue
        improvement = ref_latency - f(best, "e2e_prefill_mean_ms")
        out.append(
            {
                "reference_label": ref.get("label", ""),
                "reference_accuracy": ref_acc,
                "reference_e2e_prefill_mean_ms": ref_latency,
                "dominating_search_key": best["key"],
                "search_accuracy": f(best, "global_accuracy"),
                "search_e2e_prefill_mean_ms": f(best, "e2e_prefill_mean_ms"),
                "latency_improvement_ms": improvement,
                "latency_improvement_pct": improvement / ref_latency if ref_latency > 0 else 0.0,
            }
        )
    return out


def write_report(
    output_root: Path,
    rows: list[dict[str, Any]],
    searched: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    gap: list[dict[str, Any]],
    measured_reference: bool,
) -> None:
    breakthroughs = [row for row in gap if row.get("dominating_search_key")]
    max_gap = max((f(row, "latency_improvement_pct") for row in breakthroughs), default=0.0)
    lines = [
        "# FakeVLM Pareto Search Audit",
        "",
        f"- Total policies validated: {len(rows)}",
        f"- Search policies validated: {len(searched)}",
        f"- Non-dominated searched policies: {len(frontier)}",
        f"- 024 reference points: {len(reference)}",
        f"- 024 reference source: {'measured with 025 validator' if measured_reference else '024 report'}",
        f"- Reference points dominated by searched policies: {len(breakthroughs)}",
        f"- Max latency improvement at equal/higher accuracy: {max_gap * 100:.2f}%",
        "",
        "## Non-dominated searched policies",
        "",
        "| key | family | accuracy | e2e ms | counts |",
        "|---|---|---:|---:|---|",
    ]
    for row in frontier:
        counts = ", ".join(f"{name}={row.get(f'count_{name}', '')}" for name in ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4"))
        lines.append(f"| `{row['key']}` | {row.get('family', '')} | {f(row, 'global_accuracy'):.6f} | {f(row, 'e2e_prefill_mean_ms'):.3f} | {counts} |")
    lines.extend(["", "## Gap to 024 reference", "", "| reference | search key | improvement |", "|---|---|---:|"])
    for row in gap:
        lines.append(f"| {row.get('reference_label', '')} | `{row.get('dominating_search_key', '')}` | {f(row, 'latency_improvement_pct') * 100:.2f}% |")
    (output_root / "summary").mkdir(parents=True, exist_ok=True)
    (output_root / "summary" / "search_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(output_root: Path, rows: list[dict[str, Any]], frontier: list[dict[str, Any]], reference: list[dict[str, Any]]) -> None:
    complete = [row for row in rows if row.get("global_accuracy") and row.get("e2e_prefill_mean_ms")]
    if not complete:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    families = sorted({row.get("family", "") for row in complete})
    for family in families:
        subset = [row for row in complete if row.get("family", "") == family]
        ax.scatter([f(row, "e2e_prefill_mean_ms") for row in subset], [f(row, "global_accuracy") for row in subset], s=24, alpha=0.7, label=family)
    if reference:
        ax.plot([f(row, "e2e_prefill_mean_ms") for row in reference], [f(row, "fakeclue_accuracy") for row in reference], color="black", marker="o", linewidth=1.5, label="024 reference")
    if frontier:
        ax.scatter([f(row, "e2e_prefill_mean_ms") for row in frontier], [f(row, "global_accuracy") for row in frontier], marker="x", color="red", s=60, label="searched frontier")
    ax.set_xlabel("Real prefill E2E latency (ms)")
    ax.set_ylabel("FakeClue 20% accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    (output_root / "summary").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_root / "summary" / "search_speed_vs_accuracy.png", dpi=200)
    plt.close(fig)


def plot_speedup(output_root: Path, rows: list[dict[str, Any]], frontier: list[dict[str, Any]], reference: list[dict[str, Any]]) -> None:
    complete = [row for row in rows if row.get("global_accuracy") and row.get("e2e_prefill_mean_ms")]
    baseline_latency = measured_dense_latency(reference) or measured_dense_latency(complete)
    if not complete or baseline_latency <= 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    families = sorted({row.get("family", "") for row in complete})
    for family in families:
        subset = [row for row in complete if row.get("family", "") == family]
        ax.scatter([speedup(row, baseline_latency) for row in subset], [f(row, "global_accuracy") for row in subset], s=24, alpha=0.7, label=family)
    if reference:
        ax.plot([speedup(row, baseline_latency) for row in reference], [f(row, "fakeclue_accuracy") for row in reference], color="black", marker="o", linewidth=1.5, label="024 reference")
    if frontier:
        ax.scatter([speedup(row, baseline_latency) for row in frontier], [f(row, "global_accuracy") for row in frontier], marker="x", color="red", s=60, label="searched frontier")
    ax.axvline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xlabel("Real prefill E2E speedup vs dense BF16 (x)")
    ax.set_ylabel("FakeClue 20% accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    (output_root / "summary").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_root / "summary" / "search_speedup_vs_accuracy.png", dpi=200)
    plt.close(fig)


def plot_speedup_with_uniform(
    output_root: Path,
    rows: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    uniform: list[dict[str, Any]],
) -> None:
    complete = [row for row in rows if row.get("global_accuracy") and row.get("e2e_prefill_mean_ms")]
    baseline_latency = measured_dense_latency(reference) or measured_dense_latency(complete)
    if not complete or baseline_latency <= 0:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    families = sorted({row.get("family", "") for row in complete})
    for family in families:
        subset = [row for row in complete if row.get("family", "") == family]
        ax.scatter([speedup(row, baseline_latency) for row in subset], [f(row, "global_accuracy") for row in subset], s=24, alpha=0.65, label=family)
    if reference:
        ax.plot([speedup(row, baseline_latency) for row in reference], [f(row, "fakeclue_accuracy") for row in reference], color="black", marker="o", linewidth=1.8, label="ours")
    if uniform:
        xs = [speedup(row, baseline_latency) for row in uniform]
        ys = [f(row, "fakeclue_accuracy") for row in uniform]
        ax.scatter(xs, ys, marker="s", color="#7b3294", edgecolor="white", linewidth=0.6, s=54, label="uniform baselines", zorder=4)
        for x, y, row in zip(xs, ys, uniform):
            label = short_uniform_label(row.get("label", ""))
            offset, va = uniform_label_offset(label)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8, color="#5e2a7e", va=va)
    if frontier:
        ax.scatter([speedup(row, baseline_latency) for row in frontier], [f(row, "global_accuracy") for row in frontier], marker="x", color="red", s=64, label="searched frontier", zorder=5)
    ax.axvline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xlabel("Real prefill E2E speedup vs dense BF16 (x)")
    ax.set_ylabel("FakeClue accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.85, ncol=2)
    (output_root / "summary").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_root / "summary" / "search_speedup_vs_accuracy_with_uniform.png", dpi=220)
    fig.savefig(output_root / "summary" / "search_speedup_vs_accuracy_with_uniform.pdf")
    plt.close(fig)


def uniform_024_rows(batch_size: int) -> list[dict[str, Any]]:
    path = SOURCE_024_ROOT / "report" / "final_fakevlm_report_refined_sparse_bf16_v4.csv"
    if not path.exists():
        return []
    rows = [row for row in read_csv(path) if row.get("row_type") == "uniform" and int(f(row, "batch_size")) == batch_size]
    return sorted(rows, key=lambda row: f(row, "point_index"))


def short_uniform_label(label: str) -> str:
    return (
        label.replace("Uniform ", "")
        .replace("dense BF16", "dense BF16")
        .replace("dense NVFP4", "dense NVFP4")
        .replace("sparse BF16", "sparse BF16")
        .replace("sparse NVFP4", "sparse NVFP4")
    )


def uniform_label_offset(label: str) -> tuple[tuple[int, int], str]:
    if label in {"dense BF16", "dense NVFP4"}:
        return (6, -12), "top"
    if label == "sparse BF16":
        return (6, 8), "bottom"
    return (6, 6), "bottom"


def measured_dense_latency(rows: list[dict[str, Any]]) -> float:
    dense_rows = [
        row
        for row in rows
        if int(f(row, "count_dense_bf16")) == 224
        and int(f(row, "count_dense_nvfp4")) == 0
        and int(f(row, "count_sparse_bf16")) == 0
        and int(f(row, "count_sparse_nvfp4")) == 0
        and f(row, "e2e_prefill_mean_ms") > 0
    ]
    if not dense_rows:
        return 0.0
    return f(dense_rows[0], "e2e_prefill_mean_ms")


def speedup(row: dict[str, Any], baseline_latency: float) -> float:
    latency = f(row, "e2e_prefill_mean_ms")
    return baseline_latency / latency if latency > 0 else 0.0


if __name__ == "__main__":
    main()
