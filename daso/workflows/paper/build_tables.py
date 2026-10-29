#!/usr/bin/env python3
"""Deterministically build every numeric table used by the final paper."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


METHOD_ORDER = ("BF16", "Dense NVFP4", "W4A16 Marlin", "Sparse BF16",
                "Sparse NVFP4", "DASO (Max speed)", "DASO (Balanced)")
MODEL_LATEX = {
    "llama2": r"\shortstack[c]{Llama-2\\7B-Chat}",
    "llama31": r"\shortstack[c]{Llama-3.1\\8B-Instruct}",
}


def format2(value: float | Decimal) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_selector(value: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in value.split(";") if item)


def select(path: Path, selector: str) -> dict[str, str]:
    where = parse_selector(selector)
    matches = [row for row in read_rows(path)
               if all(row.get(key) == value for key, value in where.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row in {path} for {where}, found {len(matches)}")
    return matches[0]


def prefill_cells(row: dict[str, str]) -> list[float]:
    if "arc_challenge_acc_norm" in row:
        fields = ("arc_challenge_acc_norm", "arc_easy_acc", "winogrande_acc", "mmlu_acc")
        speed = "measured_speedup"
    else:
        fields = ("arc_challenge", "arc_easy", "winogrande", "mmlu")
        speed = "speedup"
    return [100 * float(row[field]) for field in fields] + [float(row[speed])]


def decode_cells(quality: dict[str, str], speed: dict[str, str],
                 baseline: dict[str, str]) -> list[float]:
    quality_values = [float(quality[field]) for field in
                      ("cnn_dm_rougeL_percent", "dsum_rougeL_percent", "iwslt_sacre_bleu")]
    baseline_ttft = float(baseline["ttft_median_ms"])
    baseline_e2e = float(baseline["e2e_median_ms"])
    baseline_tpot = float(baseline.get("tpot_ms") or
                          (baseline_e2e - baseline_ttft) / 63)
    ttft = float(speed["ttft_median_ms"])
    e2e = float(speed["e2e_median_ms"])
    tpot = float(speed.get("tpot_ms") or (e2e - ttft) / 63)
    return quality_values + [baseline_ttft / ttft, baseline_tpot / tpot, baseline_e2e / e2e]


def main_table_values(root: Path) -> dict[tuple[str, str], list[float]]:
    values = {}
    for spec in read_rows(root / "tables/main_table_sources.csv"):
        prefill = select(root / spec["prefill_source"], spec["prefill_selector"])
        quality = select(root / spec["decode_quality_source"], spec["decode_quality_selector"])
        speed = select(root / spec["decode_speed_source"], spec["decode_speed_selector"])
        baseline = select(root / spec["decode_baseline_source"], spec["decode_baseline_selector"])
        values[spec["model"], spec["method"]] = prefill_cells(prefill) + \
            decode_cells(quality, speed, baseline)
    return values


def render_main_table(root: Path) -> str:
    values = main_table_values(root)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{1.8pt}",
             r"\begin{tabular}{@{}llccccc|cccccc@{}}", r"\toprule", "& &",
             r"\multicolumn{5}{c}{\textbf{Prefill-only}} &",
             r"\multicolumn{6}{c}{\textbf{Prefill--Decode}} \\",
             "Model & Method &", "ARC-C & ARC-E & Wino. & MMLU & TTFT &",
             r"CNN/DM & DialogSum & IWSLT'17 & TTFT & TPOT & E2E \\", r"\midrule"]
    for model_index, model in enumerate(("llama2", "llama31")):
        for method_index, method in enumerate(METHOD_ORDER):
            prefix = (rf"\multirow{{7}}{{*}}{{{MODEL_LATEX[model]}}}" if method_index == 0 else "")
            cells = " & ".join(format2(value) for value in values[model, method])
            lines.append(f"{prefix} & {method:<19} & {cells} " + r"\\")
        if model_index == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Measured downstream quality and speedups of uniform baselines and DASO on RTX~5090. Time to first token (TTFT), time per output token (TPOT), and end-to-end (E2E) speedups are relative to BF16 under the same workload. ``Balanced'' and ``Max speed'' are returned under tight and relaxed quality budgets, respectively.}",
              r"\label{tab:overall_results}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def pro6000_values(root: Path) -> dict[str, list[Decimal]]:
    rows = read_rows(root / "cross_platform_pro6000/data/selected_results.csv")
    output: dict[str, list[Decimal]] = defaultdict(list)
    for method in METHOD_ORDER:
        source_method = method.replace("DASO", "Ours")
        pre = next(row for row in rows if row["model"] == "llama2_7b_chat"
                   and row["scenario"] == "prefill_only" and row["table_role"] == source_method)
        dec = next(row for row in rows if row["model"] == "llama2_7b_chat"
                   and row["scenario"] == "prefill_decode" and row["table_role"] == source_method)
        accuracy = sum(Decimal(pre[f"metric_{index}"]) for index in range(1, 5)) / Decimal(4)
        score = sum(Decimal(dec[f"metric_{index}"]) for index in range(1, 4)) / Decimal(3)
        output[method] = [accuracy, Decimal(pre["ttft_speedup"]), score,
                          Decimal(dec["ttft_speedup"]), Decimal(dec["tpot_speedup"]),
                          Decimal(dec["speedup"])]
    return dict(output)


def render_pro6000_table(root: Path) -> str:
    values = pro6000_values(root)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lcS|cSSS@{}}",
             r"\toprule", r"& \multicolumn{2}{c|}{\textbf{Prefill-only}}",
             r"& \multicolumn{4}{c}{\textbf{Prefill--Decode}} \\",
             r"Method & Acc. & \multicolumn{1}{c|}{TTFT} & Score &",
             r"\multicolumn{1}{c}{TTFT} & \multicolumn{1}{c}{TPOT} &",
             r"\multicolumn{1}{c}{E2E} \\", r"\midrule"]
    for method in METHOD_ORDER:
        cells = " & ".join(format2(value) for value in values[method])
        lines.append(f"{method:<16} & {cells} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Results for Llama-2-7B-Chat on RTX~PRO~6000.}",
              r"\label{tab:pro6000_results}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def ablation_values(root: Path) -> list[tuple[str, str, str]]:
    rows = read_rows(root / "ablations/data/key_metrics.csv")
    lookup = {(row["category"], row["variant"]): Decimal(row["value"]) for row in rows}
    hidden = lookup["Runtime overlap", "Hidden ratio"]
    return [
        ("Kernel modeling", "Direct fit", f"{format2(100 * lookup['Kernel prediction', 'Direct fit'])}\\%"),
        ("Kernel modeling", "+ Roofline", f"{format2(100 * lookup['Kernel prediction', '+ Roofline prior'])}\\%"),
        ("Kernel modeling", "+ Shape corr.", f"{format2(100 * lookup['Kernel prediction', '+ Shape correction'])}\\%"),
        ("Quality modeling", "Local aggregation", f"{lookup['Quality prediction', 'Local aggregation']:.4f}"),
        ("Quality modeling", "+ Structure-aware", f"{lookup['Quality prediction', '+ Structure awareness']:.4f}"),
        ("Quality modeling", "+ Phase-aware", f"{lookup['Quality prediction', '+ Phase awareness']:.4f}"),
        ("Runtime deployment", "Cumulative transfer",
         f"{format2(lookup['Runtime overlap', 'Total transfer work'])}\\,ms"),
        ("Runtime deployment", "Exposed after overlap",
         f"{format2(lookup['Runtime overlap', 'Exposed copy stall'])}\\,ms ({format2(100 * hidden)}\\%)"),
    ]


def render_ablation_table(root: Path) -> str:
    rows = ablation_values(root)
    counts = {category: sum(item[0] == category for item in rows) for category, _, _ in rows}
    emitted: set[str] = set()
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lll@{}}", r"\toprule",
             r"Ablation type & Variant & Result \\", r"\midrule"]
    previous = None
    for category, variant, value in rows:
        if previous is not None and category != previous:
            lines.append(r"\midrule")
        prefix = rf"\multirow{{{counts[category]}}}{{*}}{{{category}}}" if category not in emitted else ""
        emitted.add(category)
        display = rf"\textbf{{{value}}}" if variant in {"+ Shape corr.", "+ Phase-aware"} else value
        if variant == "Exposed after overlap":
            latency, ratio = value.split(" ", 1)
            display = rf"\textbf{{{latency}}} {ratio}"
        lines.append(f"{prefix} & {variant} & {display} " + r"\\")
        previous = category
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Ablation from cost modeling to runtime deployment. Kernel modeling reports MAPE, quality modeling reports $\Delta\mathrm{NLL}$ MAE, and runtime deployment reports phase-switch transfer work and its exposed portion after overlap.}",
              r"\label{tab:modeling-ablation}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def expected_tables(root: Path) -> dict[Path, str]:
    return {
        root / "tables/main_table_final.tex": render_main_table(root),
        root / "tables/cross_platform_pro6000_final.tex": render_pro6000_table(root),
        root / "tables/ablation.tex": render_ablation_table(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/paper")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tables = expected_tables(args.root.resolve())
    mismatches = []
    for path, content in tables.items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                mismatches.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if mismatches:
        raise SystemExit("out-of-date generated tables:\n" + "\n".join(mismatches))
    print("checked" if args.check else "generated", len(tables), "tables")


if __name__ == "__main__":
    main()
