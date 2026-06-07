#!/usr/bin/env python3
"""Visualize pure prefill hybrid benchmark results as a grouped bar chart."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

models = ["Llama-2-7B", "Llama-3.1-8B", "Qwen3.5-9B"]
methods = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4", "hybrid"]

# Speedup vs dense_bf16 from module-level kernel benchmarks
# batch_size=16, input_tokens=1024, M=16384 (pure prefill)
data = {
    "Llama-2-7B":   [1.0000, 1.5266, 1.9406, 1.6824, 0.9937, 2.1945],
    "Llama-3.1-8B": [1.0000, 1.5268, 1.9444, 1.6847, 0.9983, 2.4285],
    "Qwen3.5-9B":   [1.0000, 1.4622, 1.8969, 1.6208, 0.9834, 2.2766],
}

# Absolute latency (ms) for annotation
latency = {
    "Llama-2-7B":   [908.33, 594.99, 468.08, 539.89, 914.11, 413.90],
    "Llama-3.1-8B": [984.44, 644.77, 506.28, 584.35, 986.10, 405.37],
    "Qwen3.5-9B":   [972.66, 665.19, 512.77, 600.12, 989.05, 427.24],
}

colors = ["#90a4ae", "#ef5350", "#42a5f5", "#ab47bc", "#ffa726", "#66bb6a"]
hatches = ["", "", "", "", "", "//"]

x = np.arange(len(models))
width = 0.13
n_methods = len(methods)

fig, ax = plt.subplots(figsize=(14, 7))

for i, (method, color, hatch) in enumerate(zip(methods, colors, hatches)):
    offset = (i - n_methods / 2 + 0.5) * width
    values = [data[m][i] for m in models]
    bars = ax.bar(x + offset, values, width, label=method, color=color,
                  edgecolor="white", linewidth=0.5, hatch=hatch)
    for bar, val in zip(bars, values):
        if val >= 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.2f}x", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

# baseline line
ax.axhline(y=1.0, color="#333333", linewidth=1.5, linestyle="--", alpha=0.6)
ax.text(len(models) - 0.45, 1.02, "dense_bf16 baseline", fontsize=9,
        ha="right", va="bottom", color="#333333", style="italic")

ax.set_ylabel("Speedup vs dense_bf16", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 2.65)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

ax.legend(loc="upper center", ncol=n_methods, fontsize=8.5, framealpha=0.9,
          title=None, bbox_to_anchor=(0.5, 1.08), handletextpad=0.4, columnspacing=0.8)

ax.set_title(
    "Pure Prefill: batch_size=16, input_tokens=1024, M=16384  |  GPU: NVIDIA RTX 5090 32GB",
    fontsize=12, fontweight="bold", pad=30,
)

fig.tight_layout(pad=2)

png_path = OUT_DIR / "prefill_hybrid_comparison.png"
pdf_path = OUT_DIR / "prefill_hybrid_comparison.pdf"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# ── Second chart: absolute latency comparison ──
fig2, ax2 = plt.subplots(figsize=(14, 7))

for i, (method, color, hatch) in enumerate(zip(methods, colors, hatches)):
    offset = (i - n_methods / 2 + 0.5) * width
    values = [latency[m][i] for m in models]
    bars = ax2.bar(x + offset, values, width, label=method, color=color,
                   edgecolor="white", linewidth=0.5, hatch=hatch)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{val:.0f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

ax2.set_ylabel("Prefill Latency (ms)", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11)
ax2.grid(axis="y", alpha=0.3, linewidth=0.5)

ax2.legend(loc="upper center", ncol=n_methods, fontsize=8.5, framealpha=0.9,
           title=None, bbox_to_anchor=(0.5, 1.08), handletextpad=0.4, columnspacing=0.8)

ax2.set_title(
    "Pure Prefill Latency: batch_size=16, input_tokens=1024, M=16384  |  GPU: NVIDIA RTX 5090 32GB",
    fontsize=12, fontweight="bold", pad=30,
)

fig2.tight_layout(pad=2)

png_path2 = OUT_DIR / "prefill_hybrid_latency.png"
pdf_path2 = OUT_DIR / "prefill_hybrid_latency.pdf"
fig2.savefig(png_path2, dpi=150, bbox_inches="tight")
fig2.savefig(pdf_path2, dpi=150, bbox_inches="tight")
plt.close(fig2)

print(f"Saved: {png_path2}")
print(f"Saved: {pdf_path2}")
