"""Export all six fixed-transfer workloads with explicit restoration/quality scope."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'runs/019/results'


def main():
    rows=list(csv.DictReader((SOURCE/'workloads.csv').open()))
    order=['center','single','short','long','decode256','single256']
    methods=[('SmoothQuant INT8','SQ-INT8','#65789b'),('FP8 W8A8','FP8','#ae8f68'),
        ('W4A16 NVFP4','W4A16','#699d61'),('HAWQ-FP4 r=.80','HAWQ r=.80','#9272b4'),
        ('HAWQ-FP4 hardware minimum','Common HW min','#ca864b'),
        ('DASO-8 exact/validation epsilon=0.04','DASO-8, eps=.04','#20659b')]
    values={(r['method'],r['workload']):r for r in rows}
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    labels=['B8 / 2048 / 64','B1 / 2048 / 128','B8 / 512 / 128',
        'B8 / 4096 / 64','B8 / 512 / 256','B1 / 512 / 256']
    for method,label,color in methods:
        local=[values[method,w] for w in order]
        axes[0].plot(range(6),[float(r['e2e_speedup']) for r in local],marker='o',ms=4,
            lw=2 if label.startswith('DASO') else 1.2,color=color,label=label)
        if method in [m[0] for m in methods[2:]]:
            axes[1].plot(range(6),[float(r['cycle_speedup']) for r in local],marker='o',ms=4,
                lw=2 if label.startswith('DASO') else 1.2,color=color,label=label)
        if method in [m[0] for m in methods[3:]]:
            axes[2].errorbar(range(6),[float(r['delta_nll']) for r in local],
                yerr=[2*float(r['paired_se']) for r in local],marker='o',ms=4,
                lw=1.2,capsize=2,color=color,label=label)
    for ax,title in zip(axes,['E2E speedup','Cycle speedup (includes restore)','Decode quality: mean ± 2 SE']):
        ax.set_xticks(range(6),labels,rotation=40,ha='right',fontsize=8)
        ax.set_title(title);ax.grid(alpha=.2);ax.set_xlabel('Batch / prompt length / output length')
    for ax in axes[:2]:
        ax.axhline(1,color='0.4',ls='--',lw=.8);ax.set_ylabel('Speedup over own-round BF16')
    axes[0].legend(fontsize=7,loc='lower left',ncol=2)
    axes[1].legend(fontsize=7,loc='lower left')
    axes[2].axhline(0,color='0.5',lw=.8)
    axes[2].axhline(.04,color='#bb4545',ls='--',lw=1,label='eps=.04')
    axes[2].set_ylabel('Delta NLL (nats/token)');axes[2].legend(fontsize=7,loc='lower right')
    fig.tight_layout()
    for extension in ['pdf','svg','png']:
        fig.savefig(ROOT/f'qwen019_workloads.{extension}',bbox_inches='tight',dpi=180)
    plt.close(fig)
    print('Exported six-load E2E/cycle/NLL figure PDF/SVG/PNG.')


if __name__=='__main__':main()
