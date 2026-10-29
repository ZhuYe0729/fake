"""Plot verified measured frontiers from the completed comparison table."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=Path(__file__).resolve().parents[1]


def frontier(points):
    output=[];best=float('-inf')
    for x,y in sorted(points,key=lambda point:(point[0],-point[1])):
        if y>best:output.append((x,y));best=y
    return output


def main():
    assert json.loads((RUN/'results/progress.json').read_text())['complete']
    with (RUN/'results/summary_final.csv').open() as handle:rows=list(csv.DictReader(handle))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    families=[('HAWQ-v3 allocator INT4','HAWQ allocator, INT4','#377eb8','o'),
        ('HAWQ-v3 allocator FP4 control','HAWQ allocator, FP4 control','#984ea3','s'),
        ('DASO','DASO','#e41a1c','D')]
    names={'uniform_dense_bf16':'BF16','uniform_dense_nvfp4':'Dense FP4',
        'uniform_sparse_bf16':'Sparse BF16','uniform_sparse_nvfp4':'Sparse FP4',
        'uniform_w4a16_ours':'NVFP4 W4A16','uniform_int8':'INT8','uniform_marlin_int4':'INT4 W4A16',
        'phase_uniform_nvfp4_marlin':'Phase-uniform hybrid'}
    for axis,scenario,metric,title in zip(axes,['prefill_only','prefill_decode'],
        ['ttft_speedup','e2e_speedup'],['Prefill: B=8, S=2048','Prefill + decode: B=8, S=2048, O=64'],strict=True):
        for family,label,color,marker in families:
            points=[(float(row['measured_delta_nll']),float(row[metric])) for row in rows
                    if row['scenario']==scenario and row['family']==family]
            axis.scatter(*zip(*points),color=color,marker=marker,s=32,label=label,zorder=3)
            line=frontier(points)
            axis.plot(*zip(*line),color=color,alpha=.65,linewidth=1)
        for row in rows:
            if row['scenario']!=scenario or row['policy_id'] not in names:continue
            x,y=float(row['measured_delta_nll']),float(row[metric])
            axis.scatter([x],[y],c='#444444',marker='x',s=35,zorder=4)
            axis.annotate(names[row['policy_id']],(x,y),xytext=(4,4),
                          textcoords='offset points',fontsize=7)
        axis.axhline(1,color='gray',linestyle=':',linewidth=1)
        axis.axvline(0,color='gray',linestyle=':',linewidth=1)
        axis.set(title=title,xlabel='Measured test ΔNLL (nats/token)',
                 ylabel='Measured speedup vs BF16 (higher is faster)')
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    output=RUN/'results/figures';output.mkdir(exist_ok=True)
    for suffix in ['png','pdf','svg']:fig.savefig(output/f'measured_quality_latency.{suffix}',dpi=220)
    plt.close(fig)


if __name__=='__main__':main()
