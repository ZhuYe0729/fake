#!/usr/bin/env python3
"""Merge full ARC-Challenge quality with the measured prefill-only speed axis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SPEED = ROOT / 'report/actual_nll_speed_summary.csv'
FULL = ROOT / 'arc_challenge/full'
REPORT = ROOT / 'arc_challenge/report'
LABELS = {
    'dense_bf16': ('uniform', 'dense_bf16'),
    'dense_nvfp4': ('uniform', 'dense_nvfp4'),
    'marlin_nvfp4': ('uniform', 'marlin_nvfp4'),
    'sparse_bf16': ('uniform', 'sparse_bf16'),
    'sparse_nvfp4': ('uniform', 'sparse_nvfp4'),
    'ours_point_004': ('ours', 'ours_4'),
    'ours_point_006': ('ours', 'ours_6'),
    'ours_point_008': ('ours', 'ours_8'),
    'ours_point_009': ('ours', 'ours_9'),
    'ours_point_011': ('ours', 'ours_11'),
    'ours_point_012': ('ours', 'ours_12'),
    'ours_point_013': ('ours', 'ours_13'),
    'ours_point_014': ('ours', 'ours_14'),
    'ours_point_015': ('ours', 'ours_15'),
    'ours_point_016': ('ours', 'ours_16'),
}


def main() -> None:
    speed = {(r['family'], r['label']): r for r in csv.DictReader(SPEED.open())}
    rows = []
    for path in sorted(FULL.glob('*.json')):
        result = json.loads(path.read_text())
        family, speed_label = LABELS[result['label']]
        measured = speed[(family, speed_label)]
        rows.append({
            'label': result['label'], 'family': family,
            'speedup_vs_dense': float(measured['speedup_vs_dense']),
            'e2e_median_ms': float(measured['e2e_median_ms']),
            'arc_acc_norm': float(result['acc_norm']),
            'arc_acc': float(result['acc']),
            'sample_len': result['raw_metrics']['sample_len'],
        })
    for row in rows:
        row['pareto_kept'] = not any(
            other['speedup_vs_dense'] >= row['speedup_vs_dense']
            and other['arc_acc_norm'] >= row['arc_acc_norm']
            and (other['speedup_vs_dense'] > row['speedup_vs_dense']
                 or other['arc_acc_norm'] > row['arc_acc_norm'])
            for other in rows
        )
    REPORT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (REPORT / 'arc_challenge_speed_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['speedup_vs_dense']))

    plt.figure(figsize=(10, 6.5), dpi=180)
    for family, color, marker, name in [
        ('uniform', '#d62728', 's', 'Uniform baselines'),
        ('ours', '#1f2937', 'o', 'Mixed Pareto policies'),
    ]:
        subset = [r for r in rows if r['family'] == family]
        plt.scatter([r['speedup_vs_dense'] for r in subset], [r['arc_acc_norm'] for r in subset],
                    color=color, marker=marker, s=90, label=name, zorder=3)
        for row in subset:
            offsets = {
                'dense_bf16': (7, 12), 'marlin_nvfp4': (-18, -38),
                'dense_nvfp4': (7, -15), 'sparse_bf16': (7, -15),
                'sparse_nvfp4': (7, -15), 'ours_point_004': (10, -18),
                'ours_point_006': (9, -38), 'ours_point_008': (8, -20),
                'ours_point_009': (7, -15), 'ours_point_011': (-68, -18),
                'ours_point_012': (8, 8), 'ours_point_013': (8, -20),
                'ours_point_014': (7, -15), 'ours_point_015': (7, -15),
                'ours_point_016': (7, -15),
            }
            plt.annotate(row['label'].replace('ours_point_', 'ours '),
                         (row['speedup_vs_dense'], row['arc_acc_norm']),
                         xytext=offsets[row['label']], textcoords='offset points', color=color, fontsize=9)
    frontier = sorted((r for r in rows if r['pareto_kept']), key=lambda r: r['speedup_vs_dense'])
    plt.plot([r['speedup_vs_dense'] for r in frontier], [r['arc_acc_norm'] for r in frontier],
             color='#1f2937', linewidth=2.6, zorder=2, label='Measured ARC Pareto frontier')
    plt.xlabel('E2E prefill speedup vs dense BF16 (b=8, input=2048)')
    plt.ylabel('ARC-Challenge acc_norm (full, 1,172 examples)')
    plt.title('Llama2-7B prefill-only: speed vs ARC-Challenge')
    plt.grid(alpha=.28)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(REPORT / 'pareto_speedup_vs_arc_challenge.png')

    zoom = [r for r in rows if r['label'] in {
        'dense_bf16', 'marlin_nvfp4', 'ours_point_004', 'ours_point_006',
        'ours_point_008', 'ours_point_009',
    }]
    plt.figure(figsize=(8.3, 5.2), dpi=180)
    for family, color, marker, name in [
        ('uniform', '#d62728', 's', 'Uniform baselines'),
        ('ours', '#1f2937', 'o', 'Mixed Pareto policies'),
    ]:
        subset = [r for r in zoom if r['family'] == family]
        plt.scatter([r['speedup_vs_dense'] for r in subset], [r['arc_acc_norm'] for r in subset],
                    color=color, marker=marker, s=92, label=name, zorder=3)
    for row, offset in zip(sorted(zoom, key=lambda r: r['speedup_vs_dense']),
                           [(6, 9), (6, -24), (6, -34), (6, 9), (6, -18), (6, -18)]):
        plt.annotate(row['label'].replace('ours_point_', 'ours '),
                     (row['speedup_vs_dense'], row['arc_acc_norm']),
                     xytext=offset, textcoords='offset points', color='#1f2937' if row['family'] == 'ours' else '#d62728', fontsize=10)
    plt.xlim(.98, 1.45)
    plt.ylim(.426, .436)
    plt.xlabel('E2E prefill speedup vs dense BF16 (b=8, input=2048)')
    plt.ylabel('ARC-Challenge acc_norm (full, 1,172 examples)')
    plt.title('High-quality prefill-only trade-off')
    plt.grid(alpha=.28)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(REPORT / 'pareto_speedup_vs_arc_challenge_high_quality.png')

    plateau = [r for r in rows if r['label'] in {
        'dense_bf16', 'marlin_nvfp4', 'dense_nvfp4', 'ours_point_004',
        'ours_point_006', 'ours_point_008', 'ours_point_009', 'ours_point_011',
        'ours_point_012', 'ours_point_013',
    }]
    plt.figure(figsize=(9.0, 5.4), dpi=180)
    for family, color, marker, name in [
        ('uniform', '#d62728', 's', 'Uniform baselines'),
        ('ours', '#1f2937', 'o', 'Mixed Pareto policies'),
    ]:
        subset = [r for r in plateau if r['family'] == family]
        plt.scatter([r['speedup_vs_dense'] for r in subset], [r['arc_acc_norm'] for r in subset],
                    color=color, marker=marker, s=92, label=name, zorder=3)
    focus_frontier = [r for r in frontier if r['label'] in {p['label'] for p in plateau}]
    plt.plot([r['speedup_vs_dense'] for r in focus_frontier], [r['arc_acc_norm'] for r in focus_frontier],
             color='#1f2937', linewidth=2.6, zorder=2, label='Measured ARC Pareto frontier')
    focus_offsets = {
        'dense_bf16': (6, 9), 'marlin_nvfp4': (6, -28), 'ours_point_004': (6, -20),
        'ours_point_006': (6, -38), 'ours_point_008': (6, -18), 'ours_point_009': (6, -20),
        'ours_point_011': (-60, -18), 'ours_point_012': (8, 8), 'ours_point_013': (8, -18),
        'dense_nvfp4': (8, -18),
    }
    for row in plateau:
        plt.annotate(row['label'].replace('ours_point_', 'ours '),
                     (row['speedup_vs_dense'], row['arc_acc_norm']), xytext=focus_offsets[row['label']],
                     textcoords='offset points', color='#1f2937' if row['family'] == 'ours' else '#d62728', fontsize=9)
    plt.xlim(.98, 1.92)
    plt.ylim(.425, .445)
    plt.xlabel('E2E prefill speedup vs dense BF16 (b=8, input=2048)')
    plt.ylabel('ARC-Challenge acc_norm (full, 1,172 examples)')
    plt.title('High-quality prefill-only Pareto trade-off')
    plt.grid(alpha=.28)
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3)
    plt.tight_layout()
    plt.savefig(REPORT / 'pareto_speedup_vs_arc_challenge_quality_plateau.png', bbox_inches='tight')
    print(f'wrote {len(rows)} rows and {len(frontier)} non-dominated points')


if __name__ == '__main__':
    main()
