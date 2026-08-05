#!/usr/bin/env python3
"""Summarize the three CUDA-event overlap diagnostics."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = [json.loads(path.read_text())
            for path in sorted((EXP / "profile").glob("sample_*.json"))]
    if len(rows) < 3:
        raise RuntimeError(f"expected at least three profile samples, got {len(rows)}")
    keys = ("total_transfer_work_ms", "exposed_copy_stall_ms",
            "hidden_transfer_work_ms", "hidden_ratio")
    summary = {f"median_{key}": statistics.median(row[key] for row in rows)
               for key in keys}
    summary["samples"] = len(rows)
    summary["raw"] = rows
    report = EXP / "report"
    report.mkdir(exist_ok=True)
    (report / "overlap_profile.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    text = f"""# CUDA-event overlap profile

| Mechanism metric | Median |
|---|---:|
| Total decode-state transfer work | {summary['median_total_transfer_work_ms']:.2f} ms |
| Exposed copy stall | {summary['median_exposed_copy_stall_ms']:.2f} ms |
| Hidden transfer work | {summary['median_hidden_transfer_work_ms']:.2f} ms |
| Hidden ratio | {100.0 * summary['median_hidden_ratio']:.2f}% |

These are medians of {len(rows)} independent CUDA-event diagnostic runs on the
same Llama3.1 `point_011` B=8/S=2048/O=64 workload. `Total transfer work` sums
the per-layer decode-state materialization intervals on the copy stream.
`Exposed copy stall` sums the first-use copy-event stalls observed on the
compute stream. Profiling is diagnostic-only and is not used for formal E2E
latency.
"""
    (report / "overlap_profile.md").write_text(text)
    tex = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{Effectiveness of asynchronous phase-switch overlap.}}
\\begin{{tabular}}{{lc}}
\\toprule
Mechanism metric & Median \\\\
\\midrule
Total transfer work & {summary['median_total_transfer_work_ms']:.2f} ms \\\\
Exposed copy stall & {summary['median_exposed_copy_stall_ms']:.2f} ms \\\\
Hidden transfer work & {summary['median_hidden_transfer_work_ms']:.2f} ms \\\\
Hidden ratio & {100.0 * summary['median_hidden_ratio']:.2f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:overlap-profile}}
\\end{{table}}
"""
    (report / "overlap_profile_table.tex").write_text(tex)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
