#!/usr/bin/env python3
"""Build the compact paper-facing overlap ablation report."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


EXP = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = list(csv.DictReader((EXP / "raw_samples.csv").open()))
    rows = [row for row in rows if row["warmup"] == "0"]
    grouped = {
        variant: [float(row["e2e_latency_ms"]) for row in rows
                  if row["variant"] == variant]
        for variant in ("synchronous_copy", "asynchronous_overlap")
    }
    if any(len(values) < 5 for values in grouped.values()):
        raise RuntimeError(f"need at least five retained samples: {grouped}")
    sync = statistics.median(grouped["synchronous_copy"])
    overlap = statistics.median(grouped["asynchronous_overlap"])
    reduction = (sync - overlap) / sync * 100.0
    speedup = sync / overlap
    by_pair: dict[int, dict[str, float]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair"]), {})[row["variant"]] = float(
            row["e2e_latency_ms"])
    paired_wins = sum(
        pair["asynchronous_overlap"] < pair["synchronous_copy"]
        for pair in by_pair.values()
    )
    summary = {
        "synchronous_copy_median_ms": sync,
        "asynchronous_overlap_median_ms": overlap,
        "overlap_latency_reduction_percent": reduction,
        "overlap_speedup_vs_synchronous": speedup,
        "samples_per_variant": len(grouped["synchronous_copy"]),
        "paired_overlap_wins": paired_wins,
        "paired_comparisons": len(by_pair),
        "raw_samples_ms": grouped,
    }
    report = EXP / "report"
    report.mkdir(exist_ok=True)
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    labels = ["Synchronous copy", "Asynchronous overlap"]
    values = [sync, overlap]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    colors = ["#9ca3af", "#2563eb"]
    ax.bar(labels, values, color=colors, width=0.58)
    for index, key in enumerate(("synchronous_copy", "asynchronous_overlap")):
        samples = grouped[key]
        offsets = [(-0.12 + 0.24 * i / max(1, len(samples) - 1))
                   for i in range(len(samples))]
        ax.scatter([index + offset for offset in offsets], samples,
                   color="#111827", s=20, zorder=3)
    ax.set_ylabel("E2E latency (ms, lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.5, max(values) * 1.025,
            f"{reduction:.1f}% lower latency", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(report / "overlap_ablation.png", dpi=220)
    fig.savefig(report / "overlap_ablation.pdf")
    plt.close(fig)

    profile_path = report / "overlap_profile.json"
    mechanism = ""
    mechanism_claim = ""
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        hidden_percent = 100.0 * profile["median_hidden_ratio"]
        mechanism = f"""
## Direct overlap profile

| Mechanism metric | Median |
|---|---:|
| Total decode-state transfer work | {profile['median_total_transfer_work_ms']:.2f} ms |
| Exposed copy stall | {profile['median_exposed_copy_stall_ms']:.2f} ms |
| Hidden transfer work | {profile['median_hidden_transfer_work_ms']:.2f} ms |
| Hidden ratio | {hidden_percent:.2f}% |

The mechanism values are medians of {profile['samples']} independent
diagnostic runs using CUDA events. They are kept separate from formal E2E
timing because profiling adds instrumentation.
"""
        mechanism_claim = (
            f" Direct CUDA-event profiling further shows that only "
            f"{profile['median_exposed_copy_stall_ms']:.2f} ms of "
            f"{profile['median_total_transfer_work_ms']:.2f} ms transfer work "
            f"is exposed, hiding {hidden_percent:.1f}%.")

    md = f"""# vLLM phase-switch overlap ablation

Fixed protocol: Llama3.1-8B-Instruct, solved max-speed `point_011`, B=8,
S=2048, O=64, BF16 KV cache, chunked prefill disabled, and the common
`phase_hetero_mytest` backend. All 128 Linear modules change method between
prefill and decode. Formal timing disables tracing and alternates both variants
on the same GPU.

| Runtime variant | Median E2E latency (ms) ↓ | Samples |
|---|---:|---:|
| Synchronous phase-boundary copy | {sync:.2f} | {len(grouped['synchronous_copy'])} |
| Asynchronous overlapped prefetch | {overlap:.2f} | {len(grouped['asynchronous_overlap'])} |

Asynchronous prefetch reduces median E2E latency by **{reduction:.2f}%**
({speedup:.3f}× relative speedup) compared with exposing all decode-weight
copies at the phase boundary, and is faster in {paired_wins}/{len(by_pair)}
paired comparisons. All samples are retained; this includes one visible async
long-tail sample. The experiment changes only
`PHASE_HETERO_SYNC_COPY`; it is a runtime ablation and does not alter the main
Pareto results.
{mechanism}

Paper-facing statement: *Overlapping decode-representation prefetch with the
remaining prefill computation reduces phase-switch latency by {reduction:.1f}%
for a fully phase-heterogeneous policy.{mechanism_claim}*
"""
    (report / "summary.md").write_text(md)

    tex = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{Ablation of asynchronous phase-switch prefetch on Llama3.1-8B-Instruct.}}
\\begin{{tabular}}{{lc}}
\\toprule
Runtime variant & E2E latency (ms) $\\downarrow$ \\\\
\\midrule
Synchronous phase-boundary copy & {sync:.2f} \\\\
Asynchronous overlapped prefetch & {overlap:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:overlap-ablation}}
\\end{{table}}
"""
    (report / "ablation_table.tex").write_text(tex)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
