#!/usr/bin/env python3
"""Merge measured Pareto closure results and render a paper-oriented figure."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
POINTS = ('point_000', 'point_002', 'point_004', 'point_006', 'point_008', 'point_009')
UNIFORM = {'p00': 'dense_bf16', 'p01': 'dense_nvfp4',
           'p02': 'sparse_bf16', 'p04': 'w4a16_ours'}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def median_speed(directory: Path) -> float:
    values = [json.loads(path.read_text())['elapsed_ms']
              for path in sorted(directory.glob('measured_*.json'))]
    if len(values) != 5:
        raise RuntimeError(f'{directory}: expected five samples, got {len(values)}')
    return statistics.median(values)


def main() -> None:
    closure = ROOT / 'closure'
    predicted = {row['policy_id']: row for row in csv_rows(ROOT / 'pareto/predicted_points.csv')}
    actual_nll = {path.stem: csv_rows(path)[0] for path in (closure / 'nll').glob('point_*.csv')}
    dense_speed = median_speed(closure / 'speed' / 'point_000')
    result = []
    for point in POINTS:
        nll = actual_nll[point]
        speed = median_speed(closure / 'speed' / point)
        pred = predicted[point]
        result.append({'kind': 'ours', 'policy_id': point,
                       'speed_median_ms': speed,
                       'speedup_vs_dense': dense_speed / speed,
                       'actual_delta_nll': float(nll['target_delta_nll']),
                       'predicted_delta_nll': float(pred['predicted_delta_nll']),
                       'predicted_speedup_vs_dense': float(pred['calibrated_speedup_vs_dense'])})

    nll_rows = {row['policy_id']: row for row in csv_rows(ROOT / 'nll/prefill_decode.csv')}
    for policy, label in UNIFORM.items():
        speed = median_speed(ROOT / 'speed_calibration_continuous085_w6/runs' / policy)
        result.append({'kind': 'uniform', 'policy_id': label,
                       'speed_median_ms': speed,
                       'speedup_vs_dense': dense_speed / speed,
                       'actual_delta_nll': float(nll_rows[policy]['target_delta_nll']),
                       'predicted_delta_nll': None,
                       'predicted_speedup_vs_dense': None})

    result.sort(key=lambda row: (row['kind'] != 'ours', row['speedup_vs_dense']))
    closure.mkdir(exist_ok=True)
    with (closure / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)
    lines = ['# Llama-3.1-8B-Instruct prefill-decode measured closure', '',
             '| kind | policy | E2E ms | speedup vs dense | actual ΔNLL | predicted ΔNLL |',
             '|---|---|---:|---:|---:|---:|']
    for row in result:
        pred = '' if row['predicted_delta_nll'] is None else f"{row['predicted_delta_nll']:.4f}"
        lines.append(f"| {row['kind']} | {row['policy_id']} | {row['speed_median_ms']:.2f} | {row['speedup_vs_dense']:.3f} | {row['actual_delta_nll']:.4f} | {pred} |")
    (closure / 'summary.md').write_text('\n'.join(lines) + '\n')

    report = closure / 'report'; report.mkdir(exist_ok=True)
    ours = [row for row in result if row['kind'] == 'ours']
    base = [row for row in result if row['kind'] == 'uniform']
    fig, ax = plt.subplots(figsize=(9.2, 6.2), layout='constrained')
    ax.plot([row['speedup_vs_dense'] for row in ours], [row['actual_delta_nll'] for row in ours],
            color='#1f2937', marker='o', markersize=8, linewidth=2.8, label='Ours: Pareto')
    visible_base = [row for row in base if row['actual_delta_nll'] <= 3.2]
    offscale_base = [row for row in base if row['actual_delta_nll'] > 3.2]
    ax.scatter([row['speedup_vs_dense'] for row in visible_base], [row['actual_delta_nll'] for row in visible_base],
               color='#dc2626', marker='s', s=125, zorder=3, label='Uniform')
    for row in ours:
        ax.annotate(row['policy_id'].replace('point_', 'P'), (row['speedup_vs_dense'], row['actual_delta_nll']),
                    xytext=(5, 6), textcoords='offset points', fontsize=9, color='#1f2937')
    for row in visible_base:
        ax.annotate(row['policy_id'], (row['speedup_vs_dense'], row['actual_delta_nll']),
                    xytext=(5, -15), textcoords='offset points', fontsize=9, color='#b91c1c')
    for row in offscale_base:
        clipped_y = 3.12
        ax.scatter([row['speedup_vs_dense']], [clipped_y], color='#dc2626', marker='^', s=125, zorder=3)
        ax.annotate(f"{row['policy_id']}\nΔNLL={row['actual_delta_nll']:.1f} (off-scale)",
                    (row['speedup_vs_dense'], clipped_y), xytext=(8, -2), textcoords='offset points',
                    fontsize=9, color='#b91c1c', va='top')
    ax.set_xlabel('Measured continuous prefill-decode speedup vs dense BF16', fontsize=12)
    ax.set_ylabel('Measured WikiText phase objective ΔNLL', fontsize=12)
    ax.set_title('Llama-3.1-8B-Instruct: measured Pareto closure', fontsize=15)
    ax.set_ylim(-.25, 3.35)
    ax.grid(alpha=.28)
    ax.legend(frameon=True, loc='best')
    fig.savefig(report / 'pareto_measured_speed_nll.png', dpi=220)
    print(closure / 'summary.csv')


if __name__ == '__main__':
    main()
