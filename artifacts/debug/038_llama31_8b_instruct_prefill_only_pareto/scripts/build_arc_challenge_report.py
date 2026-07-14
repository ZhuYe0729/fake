#!/usr/bin/env python3
"""Join full ARC-Challenge scores to matched measured prefill E2E speed."""
from __future__ import annotations
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
MAPPING={'dense_bf16':('uniform','dense_bf16'),'dense_nvfp4':('uniform','dense_nvfp4'),'sparse_bf16':('uniform','sparse_bf16'),'sparse_nvfp4':('uniform','sparse_nvfp4'),'marlin_nvfp4':('uniform','marlin_nvfp4'),'ours_point_3':('ours','point_3'),'ours_point_5':('ours','point_5'),'ours_point_6':('ours','point_6'),'ours_point_8':('ours','point_8'),'ours_point_9':('ours','point_9'),'ours_point_11':('ours','point_11'),'ours_point_13':('ours','point_13')}
def main():
 speed={r['label']:r for r in csv.DictReader((ROOT/'report/measured_nll_speed_summary.csv').open())};rows=[]
 for label,(family,speed_label) in MAPPING.items():
  arc=json.loads((ROOT/'arc_challenge/full'/f'{label}.json').read_text());raw=arc['raw_metrics'];s=speed[speed_label]
  rows.append({'family':family,'label':label,'e2e_median_ms':float(s['e2e_median_ms']),'speedup_vs_dense':float(s['speedup_vs_dense']),'arc_acc':float(arc['acc']),'arc_acc_norm':float(arc['acc_norm']),'arc_acc_norm_stderr':float(raw['acc_norm_stderr,none']),'arc_samples':int(raw['sample_len'])})
 for row in rows:
  row['pareto_kept']=not any(other is not row and other['e2e_median_ms']<=row['e2e_median_ms'] and other['arc_acc_norm']>=row['arc_acc_norm'] and (other['e2e_median_ms']<row['e2e_median_ms'] or other['arc_acc_norm']>row['arc_acc_norm']) for other in rows)
 report=ROOT/'arc_challenge/report';report.mkdir(parents=True,exist_ok=True);fields=list(rows[0])
 with (report/'arc_challenge_speed_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 frontier=sorted((r for r in rows if r['pareto_kept']),key=lambda r:r['speedup_vs_dense'])
 with (report/'arc_challenge_union_frontier.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(frontier)
 plt.figure(figsize=(10.5,6.4));u=[r for r in rows if r['family']=='uniform'];o=[r for r in rows if r['family']=='ours']
 ours_frontier=sorted((r for r in o if not any(x is not r and x['e2e_median_ms']<=r['e2e_median_ms'] and x['arc_acc_norm']>=r['arc_acc_norm'] and (x['e2e_median_ms']<r['e2e_median_ms'] or x['arc_acc_norm']>r['arc_acc_norm']) for x in o)),key=lambda r:r['speedup_vs_dense'])
 plt.scatter([r['speedup_vs_dense'] for r in u],[r['arc_acc_norm'] for r in u],marker='s',s=115,color='#d62728',label='Uniform baselines',zorder=3)
 plt.scatter([r['speedup_vs_dense'] for r in o],[r['arc_acc_norm'] for r in o],s=70,color='#1f77b4',label='Ours: mixed policies',zorder=2)
 plt.plot([r['speedup_vs_dense'] for r in ours_frontier],[r['arc_acc_norm'] for r in ours_frontier],color='#1f77b4',marker='o',linewidth=2.8,label='Ours: measured curve',zorder=4)
 plt.plot([r['speedup_vs_dense'] for r in frontier],[r['arc_acc_norm'] for r in frontier],color='#4b5563',linestyle='--',linewidth=1.6,label='Union non-dominated envelope',zorder=1)
 for r in u:plt.annotate(r['label'],(r['speedup_vs_dense'],r['arc_acc_norm']),xytext=(5,-15),textcoords='offset points',color='#a51d1d',fontsize=9)
 for r in frontier:
  if r['family']=='ours':plt.annotate(r['label'],(r['speedup_vs_dense'],r['arc_acc_norm']),xytext=(5,7),textcoords='offset points',fontsize=9)
 plt.xlabel('E2E prefill speedup vs dense BF16');plt.ylabel('ARC-Challenge acc_norm (1,172 examples)');plt.title('Llama-3.1-8B-Instruct prefill-only: measured speed vs ARC-Challenge');plt.grid(alpha=.28);plt.legend();plt.tight_layout();plt.savefig(report/'pareto_speedup_vs_arc_challenge.png',dpi=180)
 print(report/'arc_challenge_speed_summary.csv')
if __name__=='__main__':main()
