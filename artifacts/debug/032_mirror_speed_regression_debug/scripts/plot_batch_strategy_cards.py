#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


POLICIES = {
    8: "policy_006_mlp_sparse_bf16_96.json",
    16: "policy_008_mlp_all_plus_attn_sparse_bf16_64.json",
    32: "policy_016_extreme_fastest_shape_stable.json",
}
GROUPS = (
    ("Attention q/k/v/o", {"q_proj", "k_proj", "v_proj", "o_proj"}),
    ("MLP expand gate/up", {"gate_proj", "up_proj"}),
    ("MLP reduce down", {"down_proj"}),
)
METHOD_ORDER = ("dense_bf16", "sparse_bf16", "dense_nvfp4")
METHOD_LABELS = {
    "dense_bf16": "Dense BF16",
    "sparse_bf16": "2:4 BF16",
    "dense_nvfp4": "Dense NVFP4",
}
COLORS = {
    "dense_bf16": "#7c8a9a",
    "sparse_bf16": "#c9cfd6",
    "dense_nvfp4": "#cf5542",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MIRROR batch-size strategy cards.")
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("artifacts/debug/030_mirror_global_pareto/speedaware_frontier/policies"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_strategy_cards_8_16_32.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {batch: summarize_policy(args.policy_dir / name) for batch, name in POLICIES.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(summaries, args.output)
    print(f"wrote {args.output}")


def summarize_policy(path: Path) -> dict[str, Counter[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Counter[str]] = {label: Counter() for label, _ in GROUPS}
    for module in data["modules"]:
        typ = str(module["module_type"])
        method = str(module["selected_method"])
        for label, types in GROUPS:
            if typ in types:
                out[label][method] += 1
                break
    return out


def plot(summaries: dict[int, dict[str, Counter[str]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card_x = 0.035
    card_w = 0.93
    card_h = 0.27
    gap = 0.035
    top = 0.965

    for card_idx, batch in enumerate((8, 16, 32)):
        y_top = top - card_idx * (card_h + gap)
        y = y_top - card_h
        draw_card(ax, card_x, y, card_w, card_h, batch, summaries[batch])

    draw_legend(ax, 0.57, 0.035)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def draw_card(ax: Any, x: float, y: float, w: float, h: float, batch: int, summary: dict[str, Counter[str]]) -> None:
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.7,
        edgecolor="#e3e8ee",
        facecolor="#f8fafc",
    )
    ax.add_patch(card)

    ax.text(x + 0.04, y + h - 0.075, f"Batch size {batch}", fontsize=28, fontweight="bold", color="#151b24", va="center")

    label_x = x + 0.04
    bar_x = x + 0.50
    bar_w = w - 0.57
    bar_h = 0.046
    row_ys = [y + h - 0.145, y + h - 0.205, y + h - 0.265]

    for (group_label, _), row_y in zip(GROUPS, row_ys):
        ax.text(label_x, row_y + bar_h / 2, group_label, fontsize=20, fontweight="bold", color="#2d333c", va="center")
        draw_stacked_bar(ax, bar_x, row_y, bar_w, bar_h, summary[group_label])


def draw_stacked_bar(ax: Any, x: float, y: float, w: float, h: float, counts: Counter[str]) -> None:
    total = sum(counts.values())
    bg = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=0,
        facecolor="#edf1f5",
    )
    ax.add_patch(bg)

    cursor = x
    segments: list[tuple[str, float, int]] = []
    for method in METHOD_ORDER:
        count = counts.get(method, 0)
        if not count:
            continue
        seg_w = w * count / total
        segments.append((method, seg_w, count))
        ax.add_patch(Rectangle((cursor, y), seg_w, h, linewidth=0, facecolor=COLORS[method], clip_path=bg))
        cursor += seg_w

    for method, seg_w, count in segments:
        left = x + sum(prev_w for prev_method, prev_w, _ in segments[: segments.index((method, seg_w, count))])
        if len(segments) > 1 and seg_w / w < 0.35:
            label = f"{count}/{total}"
        else:
            label = METHOD_LABELS[method]
            if len(segments) > 1:
                label = f"{label} {count}/{total}"
        if seg_w / w >= 0.22:
            color = "white" if method in {"dense_bf16", "dense_nvfp4"} else "#26313d"
            ax.text(left + seg_w / 2, y + h / 2, label, fontsize=11.5, fontweight="bold", color=color, ha="center", va="center")
        elif count > 0:
            ax.text(left + seg_w + 0.006, y + h / 2, f"{METHOD_LABELS[method]} x{count}", fontsize=9, color="#26313d", va="center")


def draw_legend(ax: Any, x: float, y: float) -> None:
    cursor = x
    for method in METHOD_ORDER:
        ax.add_patch(Rectangle((cursor, y), 0.025, 0.018, linewidth=0, facecolor=COLORS[method]))
        ax.text(cursor + 0.032, y + 0.009, METHOD_LABELS[method], fontsize=10.5, color="#26313d", va="center")
        cursor += 0.14


if __name__ == "__main__":
    main()
