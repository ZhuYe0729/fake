#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_dialogsum import DEBUG_ROOT, PARETO_ROOT, UNIFORM_METHODS, f, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DialogSum quality and measured normal02 speed.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--pareto-root", type=Path, default=PARETO_ROOT)
    parser.add_argument("--strict", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / "summary" / "dialogsum_pareto"
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = load_quality(args.output_root)
    speed = load_speed(args.output_root, args.pareto_root)
    method_cost = load_method_cost(args.pareto_root)
    rows = join_rows(quality, speed, method_cost)
    validate_complete(rows)
    write_csv(out_dir / "dialogsum_pareto_summary.csv", rows)
    corr = correlations(rows)
    write_csv(out_dir / "correlations.csv", corr)
    plot_metric(rows, "conditional_nll", "DialogSum conditional NLL", out_dir / "speedup_vs_nll.png", lower_is_better=True)
    plot_metric(rows, "rougeL", "DialogSum ROUGE-L", out_dir / "speedup_vs_rougeL.png", lower_is_better=False)
    write_report(out_dir / "README.md", rows, corr)
    write_json(out_dir / "metadata.json", {"rows": len(rows)})
    print(f"wrote {out_dir}")


def load_quality(root: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "quality" / "final").glob("dialogsum_*.csv")):
        rows.extend(read_csv(path))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_key(row["kind"], row["item_id"])
        out[key] = row
    return out


def load_speed(root: Path, pareto_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    stable_paths = [
        pareto_root / "validation" / "stable_e2e_repeats" / "stable_e2e_repeats_all_points.csv",
    ]
    stable_paths.extend(sorted((pareto_root / "validation" / "stable_e2e_repeats").glob("*/stable_e2e_repeats_summary.csv")))
    for path in stable_paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            if row.get("ok_repeats", "") != "" and int(f(row, "ok_repeats")) <= 0:
                continue
            point = int(f(row, "point_index"))
            out[row_key("pareto", str(point))] = row
    for path in sorted((root / "speed" / "final").glob("uniform_e2e*.csv")):
        for row in read_csv(path):
            if row.get("e2e_status") != "ok":
                continue
            out[row_key("uniform", row["method"])] = row
    return out


def load_method_cost(pareto_root: Path) -> dict[str, dict[str, Any]]:
    path = pareto_root / "summary" / "method_cost_summary.csv"
    if not path.exists():
        return {}
    return {row["method"]: row for row in read_csv(path)}


def join_rows(
    quality: dict[str, dict[str, Any]],
    speed: dict[str, dict[str, Any]],
    method_cost: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in range(10):
        key = row_key("pareto", str(point))
        q = require(quality, key, "quality")
        s = require(speed, key, "speed")
        rows.append(
            {
                "kind": "pareto",
                "label": f"P{point}",
                "point_index": point,
                "method": "",
                "quality_cost": f(q, "quality_cost"),
                "conditional_nll": f(q, "conditional_nll"),
                "conditional_ppl": f(q, "conditional_ppl"),
                "rougeL": f(q, "rougeL"),
                "rouge1": f(q, "rouge1"),
                "rouge2": f(q, "rouge2"),
                "measured_e2e_ms": f(s, "e2e_total_mean_ms"),
                "speed_source": "stable_e2e_repeats",
                "num_samples": int(f(q, "num_samples")),
                "nll_tokens": int(f(q, "nll_tokens")),
            }
        )
    for method in UNIFORM_METHODS:
        key = row_key("uniform", method)
        q = require(quality, key, "quality")
        s = require(speed, key, "speed")
        c = method_cost.get(method, {})
        rows.append(
            {
                "kind": "uniform",
                "label": method,
                "point_index": "",
                "method": method,
                "quality_cost": f(c, "quality_sum", math.nan),
                "conditional_nll": f(q, "conditional_nll"),
                "conditional_ppl": f(q, "conditional_ppl"),
                "rougeL": f(q, "rougeL"),
                "rouge1": f(q, "rouge1"),
                "rouge2": f(q, "rouge2"),
                "measured_e2e_ms": f(s, "e2e_total_mean_ms"),
                "speed_source": "uniform_e2e_validation",
                "num_samples": int(f(q, "num_samples")),
                "nll_tokens": int(f(q, "nll_tokens")),
            }
        )
    dense = next(row for row in rows if row["kind"] == "pareto" and row["point_index"] == 0)
    dense_ms = f(dense, "measured_e2e_ms")
    dense_nll = f(dense, "conditional_nll")
    dense_rouge = f(dense, "rougeL")
    for row in rows:
        row["measured_speedup_vs_p0"] = dense_ms / f(row, "measured_e2e_ms")
        row["nll_delta_vs_p0"] = f(row, "conditional_nll") - dense_nll
        row["rougeL_delta_vs_p0"] = f(row, "rougeL") - dense_rouge
    return rows


def validate_complete(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 rows, got {len(rows)}")
    for row in rows:
        for key in ("conditional_nll", "rougeL", "measured_e2e_ms", "measured_speedup_vs_p0"):
            value = f(row, key, math.nan)
            if not math.isfinite(value) or value <= 0 and key not in {"rougeL"}:
                raise RuntimeError(f"incomplete or invalid {key} for {row['label']}: {row.get(key)}")


def correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        corr_row(rows, "conditional_nll", "rougeL"),
        corr_row(rows, "quality_cost", "conditional_nll"),
        corr_row(rows, "quality_cost", "rougeL"),
    ]


def corr_row(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    vals = [(f(row, x_key, math.nan), f(row, y_key, math.nan)) for row in rows]
    vals = [(x, y) for x, y in vals if math.isfinite(x) and math.isfinite(y)]
    xs = [x for x, _ in vals]
    ys = [y for _, y in vals]
    return {
        "x": x_key,
        "y": y_key,
        "n": len(vals),
        "pearson": pearson(xs, ys),
        "spearman": pearson(ranks(xs), ranks(ys)),
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / den_x / den_y if den_x > 0 and den_y > 0 else math.nan


def ranks(values: list[float]) -> list[float]:
    order = sorted((value, idx) for idx, value in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for _, idx in order[i:j]:
            out[idx] = rank
        i = j
    return out


def plot_metric(rows: list[dict[str, Any]], metric: str, ylabel: str, path: Path, *, lower_is_better: bool) -> None:
    pareto = [row for row in rows if row["kind"] == "pareto"]
    uniform = [row for row in rows if row["kind"] == "uniform"]
    plt.figure(figsize=(8, 5))
    plt.plot([f(row, "measured_speedup_vs_p0") for row in pareto], [f(row, metric) for row in pareto], marker="o", label="Pareto P0-P9")
    for row in pareto:
        plt.annotate(row["label"], (f(row, "measured_speedup_vs_p0"), f(row, metric)), fontsize=8)
    for row in uniform:
        plt.scatter([f(row, "measured_speedup_vs_p0")], [f(row, metric)], marker="x", s=80)
        plt.annotate(row["label"], (f(row, "measured_speedup_vs_p0"), f(row, metric)), fontsize=7)
    plt.xlabel("Measured normal02 E2E speedup vs P0 dense bf16")
    plt.ylabel(ylabel)
    plt.title(f"Llama2-7B normal02 speed vs {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if lower_is_better:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_report(path: Path, rows: list[dict[str, Any]], corr: list[dict[str, Any]]) -> None:
    pareto = [row for row in rows if row["kind"] == "pareto"]
    fastest = max(pareto, key=lambda row: f(row, "measured_speedup_vs_p0"))
    best_rouge = max(rows, key=lambda row: f(row, "rougeL"))
    lines = [
        "# Llama2-7B DialogSum Pareto Summary",
        "",
        "Quality is evaluated with real compressed runtime kernels. DialogSum conditional NLL is computed on reference summary tokens only; ROUGE-L is computed from greedy generated summaries.",
        "",
        "## Key Results",
        "",
        f"- Fastest Pareto point: {fastest['label']} at {f(fastest, 'measured_speedup_vs_p0'):.4f}x, NLL {f(fastest, 'conditional_nll'):.6f}, ROUGE-L {f(fastest, 'rougeL'):.6f}.",
        f"- Best ROUGE-L row: {best_rouge['label']} at {f(best_rouge, 'rougeL'):.6f}.",
        "",
        "## Correlations",
        "",
        "| x | y | n | pearson | spearman |",
        "|---|---|---:|---:|---:|",
    ]
    for row in corr:
        lines.append(f"| {row['x']} | {row['y']} | {row['n']} | {f(row, 'pearson', math.nan):.4f} | {f(row, 'spearman', math.nan):.4f} |")
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Speedup vs NLL](speedup_vs_nll.png)",
            "",
            "![Speedup vs ROUGE-L](speedup_vs_rougeL.png)",
            "",
            "## Data",
            "",
            "- `dialogsum_pareto_summary.csv` contains all P0-P9 and uniform points.",
            "- `correlations.csv` contains Pearson/Spearman correlations.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def row_key(kind: str, item_id: str) -> str:
    return f"{kind}:{item_id}"


def require(mapping: dict[str, dict[str, Any]], key: str, name: str) -> dict[str, Any]:
    if key not in mapping:
        raise RuntimeError(f"missing {name} row for {key}")
    return mapping[key]


if __name__ == "__main__":
    main()
