#!/usr/bin/env python3
"""Visualize hybrid E2E benchmark results as a grouped bar chart."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# Final data: speedup vs dense_bf16
# batch_size=1, input_tokens=16384, output_tokens=32
models = ["Llama-2-7B", "Llama-3.1-8B", "Qwen3.5-9B"]
methods = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4", "hybrid"]

data = {
    "Llama-2-7B":    [1.00, 0.75, 1.06, 0.70, 1.05, 1.26],
    "Llama-3.1-8B":  [1.00, 0.76, 0.97, 0.69, 0.98, 1.13],
    "Qwen3.5-9B":    [1.00, 0.82, 1.27, 0.85, 0.96, 1.27],
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
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.axhline(y=1.0, color="#333333", linewidth=1.5, linestyle="--", alpha=0.6)
ax.text(len(models) - 0.5, 1.01, "dense_bf16 baseline", fontsize=9,
        ha="right", va="bottom", color="#333333", style="italic")

ax.set_ylabel("Speedup vs dense_bf16", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.45)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

ax.legend(loc="upper center", ncol=len(methods), fontsize=8.5, framealpha=0.9,
          title=None, bbox_to_anchor=(0.5, 1.08), handletextpad=0.4, columnspacing=0.8)

ax.set_title(
    "batch_size=1, input_tokens=16384, output_tokens=32  |  GPU: NVIDIA RTX 5090 32GB",
    fontsize=12, fontweight="bold", pad=30,
)

fig.tight_layout(pad=2)

png_path = OUT_DIR / "hybrid_e2e_comparison.png"
pdf_path = OUT_DIR / "hybrid_e2e_comparison.pdf"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
