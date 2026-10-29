"""Standalone exportable diagnostics; consume the complete unfiltered CSV."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader((ROOT / 'qwen016_modeling_points.csv').open()))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for axis, scene, title in zip(axes, ['prefill_only', 'prefill_decode'], ['Prefill', 'Prefill + decode']):
        local = [r for r in rows if r['scenario'] == scene]
        x = [float(r['measured_e2e_ms']) for r in local]
        y = [float(r['predicted_e2e_ms']) for r in local]
        low, high = min(x+y)*.97, max(x+y)*1.03
        axis.plot([low,high], [low,high], color='.5', linewidth=1, label='Identity')
        axis.scatter(x, y, s=24, color='#0072B2', alpha=.75, label=f'All {len(local)} frozen points')
        axis.set(xlabel='Measured E2E (ms)', ylabel='Predicted E2E (ms)', title=title,
                 xlim=(low,high), ylim=(low,high))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    for suffix in ['pdf','svg','png']:
        fig.savefig(ROOT / f'qwen016_latency_modeling.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
