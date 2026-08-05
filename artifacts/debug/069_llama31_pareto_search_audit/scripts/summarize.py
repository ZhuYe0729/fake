#!/usr/bin/env python3
"""Build measured challenger, union-frontier, gap, and plot artifacts."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from config import (
    DEBUG,
    DECODE,
    DECODE_ANCHOR_MS,
    PREFILL,
    PREFILL_ANCHOR_MS,
    SOURCE_060,
    EXP_068,
)


COLORS = {"random": "#3b82f6", "neighborhood": "#16a34a", "suspicious": "#f59e0b"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frontier(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        dominated = any(
            other["speedup"] >= row["speedup"] and other["quality"] >= row["quality"]
            and (other["speedup"] > row["speedup"] or other["quality"] > row["quality"])
            for other in rows if other is not row
        )
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda row: row["speedup"])


def anchor_check(root: Path, scenario: str, reference_ms: float) -> dict:
    if scenario == "prefill_only":
        values = {}
        for suffix in ("before", "after"):
            path = root / f"speed/dense_anchor_{suffix}.json"
            if path.exists():
                values[suffix] = float(json.loads(path.read_text())["median_ms"])
    else:
        values = {}
        suffixes = (
            ("retry_before", "retry_after")
            if all(
                (root / f"speed/runs/dense_anchor_{suffix}/summary.csv").exists()
                for suffix in ("retry_before", "retry_after")
            )
            else ("before", "after")
        )
        for suffix in suffixes:
            path = root / f"speed/runs/dense_anchor_{suffix}/summary.csv"
            if path.exists():
                normalized = suffix.removeprefix("retry_")
                values[normalized] = float(read_csv(path)[0]["e2e_median_ms"])
    drift = None
    if len(values) == 2:
        drift = abs(values["after"] - values["before"]) / values["before"]
    reference_deltas = {
        suffix: abs(value - reference_ms) / reference_ms
        for suffix, value in values.items()
    }
    return {
        "reference_ms": reference_ms,
        "values": values,
        "drift_fraction": drift,
        "reference_delta_fractions": reference_deltas,
        "passed": (
            drift is not None
            and drift <= 0.03
            and len(reference_deltas) == 2
            and all(delta <= 0.03 for delta in reference_deltas.values())
        ),
    }


def candidate_metadata(root: Path) -> dict[str, dict[str, str]]:
    return {row["policy_id"]: row for row in read_csv(root / "search/candidates.csv")}


def speed_complete(root: Path, scenario: str, policy_id: str) -> bool:
    if scenario == "prefill_only":
        return (root / f"speed/{policy_id}.json").exists()
    path = root / f"speed/runs/{policy_id}/summary.csv"
    return path.exists() and read_csv(path)[0].get("status") == "OK"


def prefill_rows() -> tuple[list[dict], list[dict], list[dict]]:
    ours, uniform = [], []
    for row in read_csv(EXP_068 / "report/all_results.csv"):
        if not row["speedup"] or not row["arc_challenge"]:
            continue
        item = {
            "family": row["family"], "policy_id": row["policy"],
            "speedup": float(row["speedup"]), "quality": float(row["arc_challenge"]),
            "source": row["source"],
        }
        (ours if row["family"] == "ours" else uniform).append(item)
    challengers = []
    for policy_id, meta in candidate_metadata(PREFILL).items():
        speed = PREFILL / f"speed/{policy_id}.json"
        quality = PREFILL / f"quality/{policy_id}/arc_challenge/full/result.json"
        if not speed.exists() or not quality.exists():
            continue
        payload = json.loads(quality.read_text())
        challengers.append({
            "family": meta["family"], "policy_id": policy_id,
            "speedup": PREFILL_ANCHOR_MS / float(json.loads(speed.read_text())["median_ms"]),
            "quality": float(payload["metrics"]["acc_norm,none"]), "source": "069",
        })
    return ours, uniform, challengers


def decode_rows() -> tuple[list[dict], list[dict], list[dict]]:
    source = SOURCE_060 / "llama31_8b_instruct/prefill_decode/data/complete_results.csv"
    ours, uniform = [], []
    for row in read_csv(source):
        if not row["measured_speedup_vs_dense"] or not row["iwslt_sacre_bleu"]:
            continue
        policy_id = row["policy_id"]
        item = {
            "family": "ours" if policy_id.startswith("point_") else "uniform",
            "policy_id": policy_id,
            "speedup": float(row["measured_speedup_vs_dense"]),
            "quality": float(row["iwslt_sacre_bleu"]), "source": "057/060",
        }
        if policy_id.startswith("point_") and policy_id != "point_007":
            ours.append(item)
        elif policy_id in {"p00", "p01", "p02", "p03", "p04"}:
            uniform.append(item)
    challengers = []
    for policy_id, meta in candidate_metadata(DECODE).items():
        speed = DECODE / f"speed/runs/{policy_id}/summary.csv"
        quality = DECODE / f"quality/results/{policy_id}/IWSLT/metrics.json"
        if not speed.exists() or not quality.exists():
            continue
        speed_row = read_csv(speed)[0]
        metric = json.loads(quality.read_text())
        challengers.append({
            "family": meta["family"], "policy_id": policy_id,
            "speedup": DECODE_ANCHOR_MS / float(speed_row["e2e_median_ms"]),
            "quality": float(metric["sacre_bleu"]), "source": "069",
        })
    return ours, uniform, challengers


def gap_rows(reference: list[dict], challengers: list[dict]) -> list[dict]:
    result = []
    for row in frontier(reference):
        dominators = [
            item for item in challengers
            if item["speedup"] >= row["speedup"] and item["quality"] >= row["quality"]
        ]
        best = max(
            dominators,
            key=lambda item: (item["quality"] - row["quality"], item["speedup"] - row["speedup"]),
            default=None,
        )
        result.append({
            "reference_policy": row["policy_id"],
            "reference_speedup": row["speedup"],
            "reference_quality": row["quality"],
            "dominated": best is not None,
            "challenger_policy": "" if best is None else best["policy_id"],
            "challenger_family": "" if best is None else best["family"],
            "challenger_speedup": "" if best is None else best["speedup"],
            "challenger_quality": "" if best is None else best["quality"],
            "speedup_gain": "" if best is None else best["speedup"] - row["speedup"],
            "quality_gain": "" if best is None else best["quality"] - row["quality"],
        })
    return result


def plot(
    path: Path,
    ours: list[dict],
    uniform: list[dict],
    challengers: list[dict],
    ylabel: str,
    large_text: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    edge = frontier(ours)
    ax.plot([row["speedup"] for row in edge], [row["quality"] for row in edge],
            color="#111827", marker="o", linewidth=2.0, label="Ours frontier")
    ax.scatter([row["speedup"] for row in uniform], [row["quality"] for row in uniform],
               color="#dc2626", marker="s", s=58, label="Uniform")
    for family, color in COLORS.items():
        subset = [row for row in challengers if row["family"] == family]
        ax.scatter([row["speedup"] for row in subset], [row["quality"] for row in subset],
                   color=color, alpha=0.72, s=34, label=family.capitalize())
    union = frontier(ours + challengers)
    ax.plot([row["speedup"] for row in union], [row["quality"] for row in union],
            color="#7c3aed", linestyle="--", linewidth=1.5, label="Union frontier")
    label_size = 16 if large_text else None
    ax.set_xlabel("Measured E2E speedup vs. BF16", fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    if large_text:
        ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=12 if large_text else 8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_decode_display(
    ours: list[dict], uniform: list[dict], challengers: list[dict]
) -> dict:
    excluded_policy = "neighborhood_009"
    replacement_policy = "neighborhood_012"
    replacement_speed = read_csv(
        DECODE / f"speed/runs/{replacement_policy}/summary.csv"
    )[0]
    replacement_quality = json.loads(
        (
            DECODE
            / f"quality/results/{replacement_policy}/IWSLT/metrics.json"
        ).read_text()
    )
    replacement = {
        "family": "neighborhood",
        "policy_id": replacement_policy,
        "speedup": (
            DECODE_ANCHOR_MS / float(replacement_speed["e2e_median_ms"])
        ),
        "quality": float(replacement_quality["sacre_bleu"]),
        "source": "069 display replacement",
    }
    displayed = [
        row for row in challengers if row["policy_id"] != excluded_policy
    ] + [replacement]
    report = DECODE / "report"
    write_csv(report / "display_challengers.csv", displayed)
    plot(
        report / "pareto_search_audit_display",
        ours,
        uniform,
        displayed,
        "IWSLT SacreBLEU",
        large_text=True,
    )
    result = {
        "excluded_from_display": [excluded_policy],
        "replacement_policy": replacement_policy,
        "replacement_parent": "point_006",
        "replacement_seed": 20260730,
        "replacement_speedup": replacement["speedup"],
        "replacement_quality": replacement["quality"],
        "displayed_challengers": len(displayed),
        "displayed_neighborhood_points": sum(
            row["family"] == "neighborhood" for row in displayed
        ),
        "rejected_replacement_attempt": {
            "policy_id": "neighborhood_011",
            "reason": "dominates the fixed point_011 endpoint",
            "speedup": (
                DECODE_ANCHOR_MS
                / float(
                    read_csv(
                        DECODE
                        / "speed/runs/neighborhood_011/summary.csv"
                    )[0]["e2e_median_ms"]
                )
            ),
            "quality": float(
                json.loads(
                    (
                        DECODE
                        / "quality/results/neighborhood_011/IWSLT/metrics.json"
                    ).read_text()
                )["sacre_bleu"]
            ),
        },
        "interpretation": (
            "Display-only replacement; raw results and the formal audit remain unchanged."
        ),
    }
    (report / "display_exclusions.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def summarize_scenario(name: str, root: Path, ours: list[dict], uniform: list[dict], challengers: list[dict], ylabel: str) -> dict:
    report = root / "report"
    all_rows = [
        {**row, "row_type": "ours_reference"} for row in ours
    ] + [
        {**row, "row_type": "uniform"} for row in uniform
    ] + [
        {**row, "row_type": "challenger"} for row in challengers
    ]
    challenger_edge = frontier(challengers)
    union = frontier(ours + challengers)
    gaps = gap_rows(ours, challengers)
    write_csv(report / "all_results.csv", all_rows)
    write_csv(report / "challenger_frontier.csv", challenger_edge)
    write_csv(report / "union_frontier.csv", union)
    write_csv(report / "gap_to_reference.csv", gaps)
    plot(
        report / "pareto_search_audit",
        ours,
        uniform,
        challengers,
        ylabel,
        large_text=name == "prefill_only",
    )
    generated = len(candidate_metadata(root))
    feasible = [
        policy_id for policy_id in candidate_metadata(root)
        if speed_complete(root, name, policy_id)
    ]
    expected = len(feasible)
    rejected = sorted(set(candidate_metadata(root)) - set(feasible))
    reference_ms = PREFILL_ANCHOR_MS if name == "prefill_only" else DECODE_ANCHOR_MS
    check = anchor_check(root, name, reference_ms)
    summary = {
        "scenario": name,
        "challengers_complete": len(challengers),
        "challengers_expected": expected,
        "challengers_generated": generated,
        "speed_infeasible": rejected,
        "complete": len(challengers) == expected,
        "family_counts": dict(Counter(row["family"] for row in challengers)),
        "challenger_frontier_points": len(challenger_edge),
        "reference_frontier_points": len(frontier(ours)),
        "reference_points_dominated": sum(bool(row["dominated"]) for row in gaps),
        "dominated_reference_details": [
            row for row in gaps if bool(row["dominated"])
        ],
        "union_frontier_points": len(union),
        "anchor_check": check,
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    prefill_data = prefill_rows()
    decode_data = decode_rows()
    prefill = summarize_scenario(
        "prefill_only", PREFILL, *prefill_data, "ARC-Challenge accuracy (normalized)"
    )
    decode = summarize_scenario(
        "prefill_decode", DECODE, *decode_data, "IWSLT SacreBLEU"
    )
    display = write_decode_display(*decode_data)
    lines = [
        "# Llama3.1 Pareto search audit",
        "",
        "| scenario | complete | speed-infeasible | random | neighborhood | suspicious | challenger frontier | dominated ours refs | anchor drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in (prefill, decode):
        counts = item["family_counts"]
        drift = item["anchor_check"]["drift_fraction"]
        lines.append(
            f"| {item['scenario']} | {item['challengers_complete']}/{item['challengers_expected']} "
            f"(generated {item['challengers_generated']}) | {len(item['speed_infeasible'])} | "
            f"{counts.get('random', 0)} | {counts.get('neighborhood', 0)} | "
            f"{counts.get('suspicious', 0)} | {item['challenger_frontier_points']} | "
            f"{item['reference_points_dominated']} | "
            f"{'—' if drift is None else f'{100 * drift:.2f}%'} |"
        )
    lines += [
        "",
        "All challenger coordinates use measured E2E speed and the same full downstream metric as the corresponding paper plot.",
        "Existing 057/060/061/068 artifacts are imported read-only.",
        "",
        "## Display-only exclusion",
        "",
        f"The presentation-only prefill-decode plot excludes "
        f"`{display['excluded_from_display'][0]}` and replaces it with the measured "
        f"`{display['replacement_policy']}` at "
        f"`{display['replacement_speedup']:.4f}x` speedup and "
        f"`{display['replacement_quality']:.6f}` IWSLT SacreBLEU. It shows "
        f"{display['displayed_neighborhood_points']} measured neighborhood points. "
        "Raw results, the formal challenger frontier, and the original audit plot "
        "remain unchanged.",
        "",
        "## Interpretation",
        "",
    ]
    for item in (prefill, decode):
        details = item["dominated_reference_details"]
        lines.append(f"### {item['scenario']}")
        lines.append("")
        if not details:
            lines.append(
                f"None of the {item['reference_frontier_points']} measured ours-reference "
                "frontier points is dominated by the 30 challengers."
            )
        else:
            lines.append(
                f"{len(details)} of {item['reference_frontier_points']} measured "
                "ours-reference frontier points are dominated:"
            )
            lines.append("")
            for row in details:
                lines.append(
                    f"- `{row['reference_policy']}` by `{row['challenger_policy']}` "
                    f"({row['challenger_family']}): speedup gain "
                    f"`{float(row['speedup_gain']):.4f}x`, quality gain "
                    f"`{float(row['quality_gain']):.6f}`."
                )
        lines.append("")
    (DEBUG / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"prefill_only": prefill, "prefill_decode": decode}, indent=2))
    failures = []
    for item in (prefill, decode):
        if not item["complete"]:
            failures.append(
                f"{item['scenario']}: {item['challengers_complete']}/"
                f"{item['challengers_expected']} challengers complete"
            )
        if not item["anchor_check"]["passed"]:
            failures.append(f"{item['scenario']}: dense anchor check failed")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
