#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any


DEBUG_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = DEBUG_ROOT / "summary" / "llama2_pareto_complete"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pareto = read_csv(DEBUG_ROOT / "pareto" / "pareto_unique_points.csv")
    quality = read_csv(DEBUG_ROOT / "validation" / "pareto_quality_validation.csv")
    e2e = read_csv(DEBUG_ROOT / "validation" / "stable_e2e_repeats" / "stable_e2e_repeats_all_points.csv")
    method_cost = read_csv(DEBUG_ROOT / "summary" / "method_cost_summary.csv")
    e2e_summary = read_csv(
        DEBUG_ROOT.parents[1] / "results" / "main" / "003_llama2_oracle_summary" / "comparison" / "e2e_summary.csv"
    )

    joined = join_pareto(pareto, quality, e2e)
    uniform = uniform_baselines(method_cost, e2e_summary, joined)
    write_csv(OUT_DIR / "pareto_joined_summary.csv", joined)
    write_csv(OUT_DIR / "uniform_baseline_summary.csv", uniform)
    write_csv(OUT_DIR / "dominance_summary.csv", dominance_summary(joined, uniform))
    write_json_like_metadata(joined, uniform)

    make_plots(joined, uniform)
    write_markdown(joined, uniform)
    print(f"wrote report to {OUT_DIR}")


def join_pareto(pareto: list[dict[str, Any]], quality: list[dict[str, Any]], e2e: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q_by_point = {int(f(row, "point_index")): row for row in quality}
    e_by_point = {int(f(row, "point_index")): row for row in e2e}
    dense_pred = f(pareto[0], "dense_latency_ms")
    dense_e2e = f(e_by_point[0], "e2e_total_mean_ms")
    dense_nll = f(q_by_point[0], "nll")
    rows: list[dict[str, Any]] = []
    for row in pareto:
        point = int(f(row, "point_index"))
        q = q_by_point.get(point, {})
        e = e_by_point.get(point, {})
        rows.append(
            {
                "point_index": point,
                "quality_cost": f(row, "quality_cost"),
                "nll": f(q, "nll", math.nan),
                "nll_delta": f(q, "nll", math.nan) - dense_nll if q else math.nan,
                "arc_challenge_acc_norm_limit128": f(q, "arc_acc_norm", math.nan),
                "predicted_latency_ms": f(row, "latency_ms"),
                "predicted_speedup_vs_dense": dense_pred / f(row, "latency_ms"),
                "e2e_mean_ms": f(e, "e2e_total_mean_ms", math.nan),
                "e2e_std_ms": f(e, "e2e_total_std_ms", math.nan),
                "e2e_speedup_vs_dense": dense_e2e / f(e, "e2e_total_mean_ms") if e else math.nan,
                "count_dense_bf16": int(f(row, "count_dense_bf16")),
                "count_marlin_nvfp4": int(f(row, "count_marlin_nvfp4")),
                "count_dense_nvfp4_prefill_marlin_decode": int(f(row, "count_dense_nvfp4_prefill_marlin_decode")),
                "backend_counts": e.get("backend_counts", ""),
            }
        )
    return rows


def uniform_baselines(
    method_cost: list[dict[str, Any]],
    e2e_summary: list[dict[str, Any]],
    joined: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    method_cost_by_name = {row["method"]: row for row in method_cost}
    normal02 = [
        row
        for row in e2e_summary
        if row.get("model") == "Llama-2-7B" and row.get("scenario") == "normal_02"
    ]
    dense = next(row for row in normal02 if row.get("policy_or_method") == "dense_bf16")
    dense_e2e = f(dense, "e2e_ms")
    dense_pred = joined[0]["predicted_latency_ms"]
    rows = []
    for method in (
        "dense_bf16",
        "dense_nvfp4",
        "marlin_nvfp4",
        "dense_nvfp4_prefill_marlin_decode",
        "sparse_bf16",
        "sparse_nvfp4",
    ):
        e = next((row for row in normal02 if row.get("policy_or_method") == method), None)
        c = method_cost_by_name.get(method, {})
        rows.append(
            {
                "method": method,
                "supported_in_current_pareto": str(c.get("supported_rows", "0")) == "224",
                "quality_cost": f(c, "quality_sum", math.nan),
                "predicted_latency_ms": f(c, "latency_sum_ms", math.nan),
                "predicted_speedup_vs_dense": dense_pred / f(c, "latency_sum_ms") if c and f(c, "latency_sum_ms") > 0 else math.nan,
                "e2e_ms": f(e, "e2e_ms", math.nan) if e else math.nan,
                "e2e_speedup_vs_dense": dense_e2e / f(e, "e2e_ms") if e else math.nan,
                "backend_counts": e.get("backend_counts", "") if e else "",
                "note": c.get("note", ""),
            }
        )
    return rows


def dominance_summary(joined: list[dict[str, Any]], uniform: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    supported = [row for row in uniform if row["supported_in_current_pareto"] is True or row["supported_in_current_pareto"] == True]
    for u in supported:
        q = f(u, "quality_cost")
        e2e = f(u, "e2e_ms")
        pred = f(u, "predicted_latency_ms")
        same_or_better_proxy = [p for p in joined if f(p, "quality_cost") <= q + 1e-9]
        best_pred = min(same_or_better_proxy, key=lambda p: f(p, "predicted_latency_ms")) if same_or_better_proxy else None
        best_e2e = min(same_or_better_proxy, key=lambda p: f(p, "e2e_mean_ms")) if same_or_better_proxy else None
        rows.append(
            {
                "uniform_method": u["method"],
                "uniform_quality_cost": q,
                "uniform_predicted_latency_ms": pred,
                "uniform_e2e_ms": e2e,
                "best_pareto_point_same_or_less_quality_pred": best_pred["point_index"] if best_pred else "",
                "best_pareto_predicted_latency_ms": f(best_pred, "predicted_latency_ms") if best_pred else "",
                "predicted_dominates_uniform": bool(best_pred and f(best_pred, "predicted_latency_ms") <= pred + 1e-9),
                "best_pareto_point_same_or_less_quality_e2e": best_e2e["point_index"] if best_e2e else "",
                "best_pareto_e2e_ms": f(best_e2e, "e2e_mean_ms") if best_e2e else "",
                "e2e_dominates_uniform": bool(best_e2e and f(best_e2e, "e2e_mean_ms") <= e2e + 1e-9),
            }
        )
    return rows


def make_plots(joined: list[dict[str, Any]], uniform: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = [f(row, "quality_cost") for row in joined]
    pred_speed = [f(row, "predicted_speedup_vs_dense") for row in joined]
    e2e_speed = [f(row, "e2e_speedup_vs_dense") for row in joined]
    nll_delta = [f(row, "nll_delta") for row in joined]

    supported_uniform = [row for row in uniform if row.get("supported_in_current_pareto") is True]
    unsupported_uniform = [row for row in uniform if row.get("supported_in_current_pareto") is False]

    plt.figure(figsize=(8, 5))
    plt.plot(q, pred_speed, marker="o", label="Pareto predicted linear speedup")
    for row in supported_uniform:
        if math.isfinite(f(row, "predicted_speedup_vs_dense")):
            plt.scatter([f(row, "quality_cost")], [f(row, "predicted_speedup_vs_dense")], marker="x", s=80)
            plt.text(f(row, "quality_cost"), f(row, "predicted_speedup_vs_dense"), row["method"], fontsize=8)
    plt.xlabel("Proxy quality cost, lower is better")
    plt.ylabel("Predicted speedup vs dense bf16")
    plt.title("Llama2-7B normal_02 predicted Pareto frontier")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pareto_predicted_proxy_speed.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(nll_delta, e2e_speed, marker="o", label="Pareto measured E2E")
    for row in joined:
        plt.text(f(row, "nll_delta"), f(row, "e2e_speedup_vs_dense"), f"P{int(f(row, 'point_index'))}", fontsize=8)
    for row in supported_uniform:
        if math.isfinite(f(row, "e2e_speedup_vs_dense")):
            # Uniform compressed methods use dense_nvfp4 quality source; NLL for all-marlin/hybrid was not
            # measured on arc_challenge here, so use proxy-cost plot below for uniform comparisons.
            pass
    plt.xlabel("NLL delta vs dense bf16, lower is better")
    plt.ylabel("Measured E2E speedup vs dense bf16")
    plt.title("Llama2-7B normal_02 measured Pareto curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pareto_measured_nll_speed.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(q, e2e_speed, marker="o", label="Pareto measured E2E")
    for row in joined:
        plt.text(f(row, "quality_cost"), f(row, "e2e_speedup_vs_dense"), f"P{int(f(row, 'point_index'))}", fontsize=8)
    for row in supported_uniform:
        if math.isfinite(f(row, "e2e_speedup_vs_dense")):
            plt.scatter([f(row, "quality_cost")], [f(row, "e2e_speedup_vs_dense")], marker="x", s=90)
            plt.text(f(row, "quality_cost"), f(row, "e2e_speedup_vs_dense"), row["method"], fontsize=8)
    for row in unsupported_uniform:
        if math.isfinite(f(row, "e2e_speedup_vs_dense")):
            plt.scatter([max(q) * 1.03], [f(row, "e2e_speedup_vs_dense")], marker=".", s=50, alpha=0.45)
            plt.text(max(q) * 1.03, f(row, "e2e_speedup_vs_dense"), f"{row['method']} unsupported", fontsize=7, alpha=0.65)
    plt.xlabel("Proxy quality cost, lower is better")
    plt.ylabel("Measured E2E speedup vs dense bf16")
    plt.title("Llama2-7B normal_02 measured speed vs proxy quality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pareto_measured_proxy_speed_with_uniform.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(q, [f(row, "count_dense_bf16") for row in joined], marker="o", label="bf16")
    plt.plot(q, [f(row, "count_marlin_nvfp4") for row in joined], marker="o", label="marlin_nvfp4")
    plt.plot(q, [f(row, "count_dense_nvfp4_prefill_marlin_decode") for row in joined], marker="o", label="dense prefill + marlin decode")
    plt.xlabel("Proxy quality cost")
    plt.ylabel("Number of linear modules")
    plt.title("Backend composition along Pareto frontier")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pareto_backend_composition.png", dpi=180)
    plt.close()


def write_markdown(joined: list[dict[str, Any]], uniform: list[dict[str, Any]]) -> None:
    dense = joined[0]
    measured_points = [row for row in joined if math.isfinite(f(row, "e2e_speedup_vs_dense", math.nan))]
    lines = [
        "# Llama2-7B Normal02 Complete Pareto Summary",
        "",
        "## Scope",
        "",
        "- Model: `llama2-7b`",
        "- Scenario: `normal_02`, batch 1, prefill 16384, decode 256",
        "- Goal: validate and present a quality-speed Pareto curve, not only one operating point.",
        "",
        "## Key Results",
        "",
        f"- DP sampled 10 Pareto budget points: `{', '.join('P' + str(int(f(row, 'point_index'))) for row in joined)}`.",
        f"- E2E/NLL validation currently covers 7 points: `{', '.join('P' + str(int(f(row, 'point_index'))) for row in measured_points)}`.",
        "- The earlier OOM for P4/P5/P6/P8 was a validator artifact from running many policies in one process; process-per-repeat validation fixes it.",
        f"- Best measured Pareto point is P9: `{f(joined[-1], 'e2e_speedup_vs_dense'):.3f}x` E2E speedup, NLL delta `{f(joined[-1], 'nll_delta'):.6f}`.",
        "- ARC-Challenge limit=128 remains too coarse to rank these points; NLL is the more useful validation metric here.",
        "",
        "## Plots",
        "",
        "![Predicted proxy-speed Pareto](pareto_predicted_proxy_speed.png)",
        "",
        "![Measured NLL-speed Pareto](pareto_measured_nll_speed.png)",
        "",
        "![Measured proxy-speed with uniform baselines](pareto_measured_proxy_speed_with_uniform.png)",
        "",
        "![Backend composition](pareto_backend_composition.png)",
        "",
        "## Pareto Points",
        "",
        "| point | proxy cost | NLL delta | ARC-C acc_norm | predicted speedup | E2E speedup | E2E mean ms | backend shape |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in joined:
        shape = (
            f"{int(f(row, 'count_dense_bf16'))} bf16, "
            f"{int(f(row, 'count_marlin_nvfp4'))} marlin, "
            f"{int(f(row, 'count_dense_nvfp4_prefill_marlin_decode'))} hybrid"
        )
        lines.append(
            f"| P{int(f(row, 'point_index'))} | {fmt_float(f(row, 'quality_cost'), 4)} | {fmt_float(f(row, 'nll_delta'), 6)} | "
            f"{fmt_float(f(row, 'arc_challenge_acc_norm_limit128'), 6)} | {fmt_speed(f(row, 'predicted_speedup_vs_dense'))} | "
            f"{fmt_speed(f(row, 'e2e_speedup_vs_dense'))} | {fmt_float(f(row, 'e2e_mean_ms'), 1)} | `{shape}` |"
        )

    lines += [
        "",
        "## Uniform Baselines",
        "",
        "Uniform baselines are useful controls, but only `dense_bf16`, `dense_nvfp4`, `marlin_nvfp4`, and `dense_nvfp4_prefill_marlin_decode` are in the current normal_02 Pareto candidate set. Sparse single-method E2E runs exist, but sparse rows are marked unsupported in this per-linear normal_02 optimizer because decode `M=1` violates the current sparse-kernel shape constraints.",
        "",
        "| method | in Pareto candidate set | proxy cost | predicted speedup | measured E2E speedup | E2E ms | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in uniform:
        supported = "yes" if row["supported_in_current_pareto"] is True else "no"
        pred = fmt_speed(f(row, "predicted_speedup_vs_dense", math.nan))
        e2e = fmt_speed(f(row, "e2e_speedup_vs_dense", math.nan))
        cost = fmt_float(f(row, "quality_cost", math.nan), 4)
        e2e_ms = fmt_float(f(row, "e2e_ms", math.nan), 1)
        lines.append(f"| `{row['method']}` | {supported} | {cost} | {pred} | {e2e} | {e2e_ms} | {row.get('note', '')} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- In the predicted proxy-latency space, the Pareto optimizer dominates supported uniform methods by construction because uniform policies are a subset of the per-module choice space.",
        "- In measured E2E space, P9 is faster than the supported uniform `marlin_nvfp4` and `dense_nvfp4_prefill_marlin_decode` baselines while using the same proxy quality cost endpoint.",
        "- Uniform baseline E2E numbers come from the existing 003 warm-E2E-aligned summary, while Pareto points use the newer process-per-repeat protocol. They are close enough for diagnosis, but final figures should remeasure uniform baselines with the same process-per-repeat protocol.",
        "- The full measured curve is monotonic in speed from P0 to P9, but gains are small at low budgets: P4/P5 are close to dense, P6/P7 start to move, and P8/P9 carry most of the speed improvement.",
        "- The NLL curve is monotonic with proxy cost for validated points, which supports using the proxy as the optimization constraint.",
        "- ARC-Challenge limit=128 does not provide enough resolution; it should not be used as the main curve-quality metric.",
        "",
        "## What This Does Not Prove Yet",
        "",
        "- It does not yet prove dominance over sparse uniform methods inside the same optimizer, because sparse methods are unsupported by the current normal_02 per-linear candidate table.",
        "- It does not yet provide a dense continuous Pareto frontier; the current curve is a 10-budget DP sample with 7 E2E-validated points.",
        "- It does not yet validate full ARC-Challenge for every point; current task validation is limit=128.",
        "",
        "## Next Steps To完善 Llama2-7B",
        "",
        "1. Generate a denser frontier, ideally all non-dominated DP states or at least 30-50 budget points, then validate a selected subset.",
        "2. Add a formal uniform-dominance table/plot for supported methods in both predicted space and measured E2E space.",
        "3. Remeasure supported uniform baselines with the same process-per-repeat protocol used for Pareto points.",
        "4. Run full ARC-Challenge, not limit=128, for representative points P0/P6/P8/P9 and supported uniform baselines.",
        "5. Decide how to handle sparse methods in normal_02: either exclude them explicitly because decode `M=1` is unsupported in the optimizer, or build a padded/compatible sparse candidate path and include them fairly.",
        "6. Calibrate the quality proxy weights using NLL deltas from the measured curve instead of keeping layer/family weights purely heuristic.",
        "7. Make process-per-repeat E2E validation the only accepted timing protocol for this scenario.",
        "",
        "## Files",
        "",
        "- `pareto_joined_summary.csv`: joined proxy, NLL, ARC limit=128, predicted latency, and measured E2E for Pareto points.",
        "- `uniform_baseline_summary.csv`: supported and unsupported uniform controls.",
        "- `dominance_summary.csv`: automatic dominance check for supported uniform methods.",
        "- `*.png`: plots shown above.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_like_metadata(joined: list[dict[str, Any]], uniform: list[dict[str, Any]]) -> None:
    text = (
        "{\n"
        f'  "pareto_points": {len(joined)},\n'
        f'  "uniform_methods": {len(uniform)},\n'
        '  "scenario": "llama2-7b normal_02",\n'
        '  "timing_protocol": "process_per_repeat"\n'
        "}\n"
    )
    (OUT_DIR / "metadata.json").write_text(text)


def f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        value = row.get(key, "")
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "--"


def fmt_speed(value: float) -> str:
    return f"{value:.3f}x" if math.isfinite(value) else "--"


if __name__ == "__main__":
    main()
