#!/usr/bin/env python3
"""Build a measured prefill-only NLL/speed table and Pareto figure."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OLD_POINTS = {0, 4, 8, 12, 16}
NEW_POINTS = {1, 3, 6, 9, 11, 13, 15}
NLL_NOISE_FLOOR = 1e-3


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def nondominated(rows: list[dict[str, object]]) -> None:
    for row in rows:
        q, t = float(row["delta_nll_for_pareto"]), float(row["e2e_median_ms"])
        row["pareto_kept"] = not any(
            other is not row
            and float(other["delta_nll_for_pareto"]) <= q
            and float(other["e2e_median_ms"]) <= t
            and (float(other["delta_nll_for_pareto"]) < q or float(other["e2e_median_ms"]) < t)
            for other in rows
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    old = root.parent / "034_llama2_7b_chat_wikitext_pareto_solver"
    quality = root.parent / "033_llama2_7b_chat_wikitext_phase_nll_proxy"
    speeds = {int(r["point_index"]): r for r in read(root / "e2e_calibration.csv")}
    rows: list[dict[str, object]] = []
    for point in sorted(OLD_POINTS | NEW_POINTS):
        nll_path = (old / f"validation/prefill_only/nll_point_{point}.csv") if point in OLD_POINTS else (root / f"actual_nll/point_{point}.csv")
        nll = read(nll_path)[0]
        raw = float(nll["target_delta_nll"])
        rows.append({
            "family": "ours", "label": f"ours_{point}", "point_index": point,
            "nll_source": "034_existing" if point in OLD_POINTS else "037_new",
            "raw_delta_nll": raw,
            "delta_nll_for_pareto": max(0.0, raw) if abs(raw) < NLL_NOISE_FLOOR else raw,
            "e2e_median_ms": float(speeds[point]["e2e_median_ms"]),
            "speed_samples": 5,
        })
    nll = {r["policy_id"]: r for r in read(quality / "nll/prefill_only.csv")}
    speed = {r["method"]: r for r in read(root.parents[1] / "exports/vllm/baselines/llama2-7b-chat/results/summary/speed_summary.csv") if r["scenario"] == "prefill_only"}
    refs = {"dense_bf16": "p00", "dense_nvfp4": "p01", "sparse_bf16": "p02", "sparse_nvfp4": "p03", "marlin_nvfp4": "p04"}
    for method, policy in refs.items():
        raw = float(nll[policy]["target_delta_nll"])
        rows.append({
            "family": "uniform", "label": method, "point_index": "",
            "nll_source": "033_uniform", "raw_delta_nll": raw,
            "delta_nll_for_pareto": max(0.0, raw) if abs(raw) < NLL_NOISE_FLOOR else raw,
            "e2e_median_ms": float(speed[method]["e2e_median_ms"]),
            "speed_samples": "baseline",
        })
    dense = next(float(r["e2e_median_ms"]) for r in rows if r["label"] == "dense_bf16")
    for row in rows:
        row["speedup_vs_dense"] = dense / float(row["e2e_median_ms"])
    nondominated(rows)
    report = root / "report"
    report.mkdir(exist_ok=True)
    out = report / "actual_nll_speed_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    frontier = sorted((r for r in rows if r["pareto_kept"]), key=lambda r: float(r["speedup_vs_dense"]))
    with (report / "actual_nll_measured_frontier.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(frontier)

    plt.figure(figsize=(10.5, 6.4))
    ours = [r for r in rows if r["family"] == "ours"]
    uniform = [r for r in rows if r["family"] == "uniform"]
    plt.scatter([r["speedup_vs_dense"] for r in uniform], [r["delta_nll_for_pareto"] for r in uniform], marker="s", s=110, color="#d62728", label="Uniform baselines", zorder=3)
    plt.scatter([r["speedup_vs_dense"] for r in ours], [r["delta_nll_for_pareto"] for r in ours], marker="o", s=62, color="#7f8c8d", alpha=.75, label="Mixed policies", zorder=2)
    plt.plot([r["speedup_vs_dense"] for r in frontier], [r["delta_nll_for_pareto"] for r in frontier], color="#1f2937", linewidth=2.8, marker="o", markersize=7, label="Measured Pareto frontier", zorder=4)
    for r in uniform:
        plt.annotate(str(r["label"]), (r["speedup_vs_dense"], r["delta_nll_for_pareto"]), xytext=(7, -15), textcoords="offset points", color="#a51d1d", fontsize=10)
    for r in frontier:
        if r["family"] == "ours":
            plt.annotate(f"ours {r['point_index']}", (r["speedup_vs_dense"], r["delta_nll_for_pareto"]), xytext=(5, 8), textcoords="offset points", fontsize=9)
    plt.xlabel("E2E prefill speedup vs dense BF16")
    plt.ylabel("WikiText ΔNLL (100 blocks; lower is better)")
    plt.title("Llama2-7B prefill-only: measured speed vs measured WikiText NLL")
    plt.grid(alpha=.28)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(report / "pareto_speedup_vs_actual_wikitext_nll.png", dpi=180)
    print(out)


if __name__ == "__main__":
    main()
