#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = SCRIPT_DIR.parents[0]

METHOD_COLORS = {
    "2:4 BF16": "#70859a",
    "2:4 W4A4": "#d84b35",
    "Hybrid: W4A4 -> W4A16": "#526374",
}

PANELS = [
    {
        "title": "Prefill-heavy",
        "rows": [
            ("Attention q/k/v/o", "2:4 BF16"),
            ("MLP expand gate/up", "2:4 W4A4"),
            ("MLP reduce down", "2:4 BF16"),
        ],
    },
    {
        "title": "Balanced prefill + decode",
        "rows": [("All language linear layers", "Hybrid: W4A4 -> W4A16")],
    },
    {
        "title": "Decode-heavy",
        "rows": [("All language linear layers", "Hybrid: W4A4 -> W4A16")],
    },
]


def main() -> None:
    cache_dir = Path("/tmp/cospaq_matplotlib_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 7.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.065, 0.92, "Workload-Aware Hybrid Policy", fontsize=20, fontweight="bold", color="#171b22")

    panel_specs = [(0.54, 0.245), (0.365, 0.115), (0.225, 0.115)]
    for panel, (y, h) in zip(PANELS, panel_specs):
        draw_panel(ax, FancyBboxPatch, Rectangle, panel, 0.06, y, 0.88, h)

    draw_callout(ax, FancyBboxPatch)
    draw_legend(ax, Rectangle)

    output = DEBUG_ROOT / "summary" / "fakevlm_cross_workload_v4_policy_case"
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.svg')}")


def draw_panel(ax, FancyBboxPatch, Rectangle, panel: dict, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            facecolor="#f8f9fb",
            edgecolor="#dde3ea",
            linewidth=1.2,
        )
    )
    row_count = len(panel["rows"])
    title_size = 17 if row_count > 1 else 15.5
    ax.text(x + 0.035, y + h - 0.045, panel["title"], fontsize=title_size, fontweight="bold", color="#1c222b")

    row_top = y + h - (0.112 if row_count > 1 else 0.083)
    row_gap = 0.043
    for index, (layer, method) in enumerate(panel["rows"]):
        row_y = row_top - index * row_gap
        ax.text(x + 0.035, row_y, layer, fontsize=11.2, fontweight="bold", color="#303741", va="center")
        chip_w = 0.40 if row_count == 1 else 0.365
        draw_method_chip(ax, FancyBboxPatch, Rectangle, x + 0.45, row_y - 0.019, chip_w, 0.038, method)


def draw_method_chip(ax, FancyBboxPatch, Rectangle, x: float, y: float, w: float, h: float, method: str) -> None:
    color = METHOD_COLORS[method]
    if method == "Hybrid: W4A4 -> W4A16":
        left_w = w * 0.48
        right_w = w - left_w
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.004,rounding_size=0.014",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.add_patch(Rectangle((x, y), left_w, h, facecolor="#d84b35", edgecolor="none", alpha=0.92))
        ax.add_patch(Rectangle((x + left_w, y), right_w, h, facecolor="#526374", edgecolor="none", alpha=0.96))
        ax.text(x + left_w * 0.5, y + h / 2, "W4A4", ha="center", va="center", fontsize=9.0, color="white", fontweight="bold")
        ax.text(
            x + left_w + right_w * 0.5,
            y + h / 2,
            "W4A16",
            ha="center",
            va="center",
            fontsize=9.0,
            color="white",
            fontweight="bold",
        )
        ax.text(x + left_w - 0.002, y + h / 2, "->", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
        return

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.014",
            facecolor=color,
            edgecolor="none",
        )
    )
    ax.text(x + w / 2, y + h / 2, method, ha="center", va="center", fontsize=10, color="white", fontweight="bold")


def draw_callout(ax, FancyBboxPatch) -> None:
    x, y, w, h = 0.09, 0.097, 0.82, 0.033
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.016",
            facecolor="#eef2f5",
            edgecolor="#d6dde5",
            linewidth=0.8,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        "Hybrid selection is workload-aware, not one-size-fits-all",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#303741",
        fontweight="bold",
    )


def draw_legend(ax, Rectangle) -> None:
    items = [
        ("2:4 BF16", METHOD_COLORS["2:4 BF16"]),
        ("2:4 W4A4", METHOD_COLORS["2:4 W4A4"]),
        ("Hybrid: W4A4 -> W4A16", METHOD_COLORS["Hybrid: W4A4 -> W4A16"]),
    ]
    x = 0.105
    y = 0.045
    for label, color in items:
        ax.add_patch(Rectangle((x, y), 0.018, 0.012, facecolor=color, edgecolor="none"))
        ax.text(x + 0.026, y + 0.006, label, va="center", fontsize=8.8, color="#3c444f")
        x += 0.285 if label != "Hybrid: W4A4 -> W4A16" else 0.0


if __name__ == "__main__":
    main()
