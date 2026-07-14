#!/usr/bin/env python3
"""Create the Llama3 prefill-only measured union Pareto table and figure."""
from __future__ import annotations
import csv,json,statistics
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
UNIFORM={'p00':'dense_bf16','p01':'dense_nvfp4','p02':'sparse_bf16','p03':'sparse_nvfp4','p04':'marlin_nvfp4'}
POINTS=(3,5,6,7,8,9,11,13)
def one(path):return next(csv.DictReader(path.open()))
def median(runs):return statistics.median(json.loads(p.read_text())['elapsed_ms'] for p in sorted(runs.glob('measured_*.json')))
def dominated(rows):
    for row in rows:
        row['pareto_kept']=not any(other is not row and other['delta_nll']<=row['delta_nll'] and other['e2e_median_ms']<=row['e2e_median_ms'] and (other['delta_nll']<row['delta_nll'] or other['e2e_median_ms']<row['e2e_median_ms']) for other in rows)
def main():
    rows=[];cal=ROOT/'speed_calibration/runs'
    for pid,label in UNIFORM.items():
        nll=one(ROOT/'nll_shards'/f'{pid}.csv');rows.append({'family':'uniform','label':label,'point_index':'','delta_nll':float(nll['target_delta_nll']),'e2e_median_ms':median(cal/pid),'speed_samples':5,'nll_samples':100})
    for point in POINTS:
        nll=one(ROOT/'closure/nll'/f'point_{point}.csv');rows.append({'family':'ours','label':f'point_{point}','point_index':point,'delta_nll':float(nll['target_delta_nll']),'e2e_median_ms':median(ROOT/'closure/speed'/f'point_{point}'/'runs'),'speed_samples':5,'nll_samples':100})
    dense=next(r['e2e_median_ms'] for r in rows if r['label']=='dense_bf16')
    for r in rows:r['speedup_vs_dense']=dense/r['e2e_median_ms']
    dominated(rows)
    report=ROOT/'report';report.mkdir(exist_ok=True)
    fields=list(rows[0])
    with (report/'measured_nll_speed_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    frontier=sorted((r for r in rows if r['pareto_kept']),key=lambda r:r['speedup_vs_dense'])
    with (report/'measured_union_frontier.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(frontier)
    plt.figure(figsize=(10.5,6.4));u=[r for r in rows if r['family']=='uniform'];o=[r for r in rows if r['family']=='ours']
    ours_frontier=sorted((r for r in o if not any(x is not r and x['delta_nll']<=r['delta_nll'] and x['e2e_median_ms']<=r['e2e_median_ms'] and (x['delta_nll']<r['delta_nll'] or x['e2e_median_ms']<r['e2e_median_ms']) for x in o)),key=lambda r:r['speedup_vs_dense'])
    plt.scatter([r['speedup_vs_dense'] for r in u],[r['delta_nll'] for r in u],marker='s',s=115,color='#d62728',label='Uniform baselines',zorder=3)
    plt.scatter([r['speedup_vs_dense'] for r in o],[r['delta_nll'] for r in o],s=70,color='#1f77b4',label='Ours: mixed policies',zorder=2)
    plt.plot([r['speedup_vs_dense'] for r in ours_frontier],[r['delta_nll'] for r in ours_frontier],color='#1f77b4',marker='o',linewidth=2.8,label='Ours: measured curve',zorder=4)
    plt.plot([r['speedup_vs_dense'] for r in frontier],[r['delta_nll'] for r in frontier],color='#4b5563',linestyle='--',linewidth=1.6,label='Union non-dominated envelope',zorder=1)
    for r in u:plt.annotate(r['label'],(r['speedup_vs_dense'],r['delta_nll']),xytext=(5,-15),textcoords='offset points',color='#a51d1d',fontsize=9)
    for r in frontier:
        if r['family']=='ours':plt.annotate(r['label'],(r['speedup_vs_dense'],r['delta_nll']),xytext=(5,7),textcoords='offset points',fontsize=9)
    plt.xlabel('E2E prefill speedup vs dense BF16');plt.ylabel('Measured WikiText ΔNLL (100 blocks; lower is better)');plt.title('Llama-3.1-8B-Instruct prefill-only: measured speed vs NLL');plt.grid(alpha=.28);plt.legend();plt.tight_layout();plt.savefig(report/'measured_pareto_speedup_vs_nll.png',dpi=180)
    print(report/'measured_nll_speed_summary.csv')
if __name__=='__main__':main()
