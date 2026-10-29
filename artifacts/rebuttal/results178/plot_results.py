"""Export all seven core comparisons with measured quality and repeat ranges."""
from pathlib import Path
import csv
import json
import os

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049'
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'runs/049/tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

audit=json.loads((OUT/'audit.json').read_text())
assert audit['complete'] and audit['tier']==1
rows=list(csv.DictReader((OUT/'comparison.csv').open()))
rounds=list(csv.DictReader((OUT/'speed_rounds.csv').open()))
styles={'bf16':('#666666','x','BF16'),
 'pp_hawq_bal':('#c47a18','o','HAWQ balanced'),'pd_hawq_bal':('#c47a18','o','HAWQ balanced'),
 'pd_hawq_fast':('#b36016','^','HAWQ speed-oriented'),
 'pp_daso_bal':('#247cab','o','DASO balanced'),'pd_daso_bal':('#247cab','o','DASO balanced'),
 'pd_daso_fast':('#06496e','s','DASO speed-oriented'),
 'pp_shared_fast':('#555555','D','Shared prefill speed point')}
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,4,figsize=(11,5.7));handles={}
for scene,positions in [('prefill_only',list(axes[0,:3])),('prefill_decode',list(axes[1]))]:
 cases=list(dict.fromkeys(r['workload'] for r in rows if r['scenario']==scene))
 assert len(cases)==len(positions)
 for ax,case in zip(positions,cases):
  seen=set()
  for r in [r for r in rows if r['workload']==case]:
   pid=r['policy']
   if pid in seen:continue
   seen.add(pid);color,marker,label=styles[pid]
   key='ttft_speedup' if scene=='prefill_only' else 'e2e_speedup'
   values=[float(x[key]) for x in rounds if x['workload']==case and x['method']==r['method']]
   assert len(values)==3
   x=float(r[key]);y=float(r['delta_nll'])
   handle=ax.errorbar(x,y,xerr=[[x-min(values)],[max(values)-x]],yerr=float(r['two_se']),
    color=color,marker=marker,markersize=4.5,linestyle='none',elinewidth=.8,capsize=2,label=label)
   handles[label]=handle
  ax.set_title(f"B{r['batch']} / S{r['prompt']}"+(f" / O{r['output']}" if scene=='prefill_decode' else ''))
  ax.set_xlabel(('TTFT' if scene=='prefill_only' else 'E2E')+' speedup vs BF16')
  ax.set_ylabel('Measured delta NLL')
  ax.axhline(0,color='#999999',ls='--',lw=.6);ax.grid(alpha=.2)
  ax.spines[['top','right']].set_visible(False)
axes[0,3].axis('off')
order=['BF16','HAWQ balanced','HAWQ speed-oriented','DASO balanced','DASO speed-oriented','Shared prefill speed point']
axes[0,3].legend([handles[k] for k in order],order,loc='center',frameon=False,fontsize=8)
axes[0,3].text(.04,.02,'Lower NLL / higher speed is better.\nBars: NLL 2SE; speed 3-round range.\nAll frozen reported points are shown.',transform=axes[0,3].transAxes,fontsize=7.5)
fig.tight_layout(h_pad=2.2,w_pad=1.4)
for extension in ['pdf','svg','png']:fig.savefig(OUT/f'quality_speed.{extension}',dpi=220,bbox_inches='tight')
plt.close(fig)
print(json.dumps({'panels':7,'display_rows':len(rows),'unique_plotted_points':32}))
