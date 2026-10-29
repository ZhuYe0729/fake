import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=Path(__file__).resolve().parents[1]


def load(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    current=load(RUN/'results/summary_final.csv')
    old=load(RUN.parent/'001/results/summary_final.csv')
    out=RUN/'results/figures';out.mkdir(exist_ok=True)
    groups=[('DASO + 8bit',[r for r in current if r['family']=='daso8'],'#087f5b','o'),
        ('DASO 5 candidates, matched fit/cost',[r for r in current if r['family']=='daso5_measured'],'#868e96','s'),
        ('8bit uniform baselines',[r for r in current if r['policy_id'] in ['uniform_sq_int8','uniform_fp8','uniform_sq_fp8']],'#1864ab','^'),
        ('Other uniform baselines',[r for r in current if r['family']=='uniform' and r['policy_id'] not in ['uniform_sq_int8','uniform_fp8','uniform_sq_fp8']],'#495057','v'),
        ('Simple phase controls',[r for r in current if r['family']=='phase-uniform'],'#7048e8','D'),
        ('Original DASO (run001)',[r for r in old if r['family']=='DASO'],'#adb5bd','x'),
        ('HAWQ FP4 control (run001)',[r for r in old if r['family']=='HAWQ-v3 allocator FP4 control'],'#d9480f','P')]
    for zoom in [False,True]:
        fig,axes=plt.subplots(1,2,figsize=(13,5.4))
        for ax,scenario,metric,title in zip(axes,['prefill_only','prefill_decode'],
            ['ttft_speedup','e2e_speedup'],['Prefill, B8 / S2048','Prefill + decode, B8 / S2048 / O64'],strict=True):
            for label,rows,color,marker in groups:
                values=[r for r in rows if r['scenario']==scenario and r.get('measured_delta_nll') and r.get(metric)]
                if zoom:values=[r for r in values if float(r['measured_delta_nll'])<=.12]
                ax.scatter([float(r['measured_delta_nll']) for r in values],
                    [float(r[metric]) for r in values],label=label,c=color,marker=marker,s=54,alpha=.85)
            ax.axhline(1,color='black',lw=.7);ax.axvline(0,color='black',lw=.5,alpha=.3)
            ax.set(title=title,xlabel='Measured test delta NLL (nats/token)',ylabel='Measured speedup vs BF16')
            ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,-.025))
        fig.suptitle('Qwen3-8B on RTX 5090: measured quality and same-GPU latency')
        fig.tight_layout(rect=(0,.20,1,.96))
        name='quality_latency_zoom' if zoom else 'quality_latency'
        for suffix in ['png','pdf','svg']:fig.savefig(out/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    baselines=[r for r in current if r['family']=='uniform' and r['scenario']=='prefill_only']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    names=[r['policy_id'].removeprefix('uniform_') for r in baselines]
    for ax,key,title in zip(axes,['ttft_speedup','e2e_speedup'],['Prefill speedup','Prefill + decode speedup'],strict=True):
        ax.bar(names,[float(r[key]) for r in baselines],color=['#087f5b' if name in ['sq_int8','fp8','sq_fp8'] else '#868e96' for name in names])
        ax.axhline(1,c='black',lw=.8);ax.set_title(title);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'baseline_speed.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
