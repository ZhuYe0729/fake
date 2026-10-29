"""Export paired quality evidence and fixed-LUT prediction figures."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9})


def export(fig,name):
    for extension in ['pdf','svg','png']:
        fig.savefig(ROOT/f'{name}.{extension}',bbox_inches='tight',dpi=180)
    plt.close(fig)


def main():
    rows=list(csv.DictReader((ROOT/'llama_matched_allocators.csv').open()))
    fig,ax=plt.subplots(figsize=(8,5.8))
    labels=[]
    for i,row in enumerate(rows):
        label=('Llama-3.1' if row['run']=='017' else 'Llama-2')+' / '+('PP' if row['scenario']=='prefill_only' else 'P+D')
        label+=' / '+('same op' if row['phase_mode']=='same_operator' else 'phase pair')+f' / r={float(row["latency_fraction"]):.2f}'
        labels.append(label)
        ax.errorbar(float(row['paired_daso_minus_hawq']),i,xerr=float(row['paired_two_se']),
            fmt='o',color='#245f91' if row['run']=='017' else '#c07825',capsize=3)
    ax.axvline(0,color='0.3',lw=1)
    ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis()
    ax.set_xlabel('Paired test NLL difference: DASO minus HAWQ (mean ± 2 SE)')
    ax.set_title('Identical candidates, latency cap and MILP; 64 shared test windows')
    ax.grid(axis='x',alpha=.2)
    export(fig,'llama_matched_paired_nll')
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,rid,title in zip(axes,['016','017','018'],['Qwen3-8B','Llama-3.1-8B','Llama-2-7B']):
        filename='qwen016_modeling_points.csv' if rid=='016' else f'llama{rid}_modeling_points.csv'
        rows=list(csv.DictReader((ROOT/filename).open()))
        rows=[r for r in rows if r['measured_e2e_ms']]
        for scene,color,label in [('prefill_only','#245f91','Prefill'),('prefill_decode','#c07825','P+D')]:
            subset=[r for r in rows if r['scenario']==scene]
            ax.scatter([float(r['measured_e2e_ms']) for r in subset],
                [float(r['predicted_e2e_ms']) for r in subset],s=20,alpha=.75,color=color,label=f'{label} (n={len(subset)})')
        low=min(min(float(r[k]) for k in ['measured_e2e_ms','predicted_e2e_ms']) for r in rows)*.95
        high=max(max(float(r[k]) for k in ['measured_e2e_ms','predicted_e2e_ms']) for r in rows)*1.04
        ax.plot([low,high],[low,high],color='0.4',ls='--',lw=1)
        ax.set(xlim=(low,high),ylim=(low,high),title=title,xlabel='Measured E2E (ms)',ylabel='Predicted E2E (ms)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout()
    export(fig,'three_model_lut_prediction')
    print('Exported two figure sets, PDF/SVG/PNG.')


if __name__=='__main__':main()
