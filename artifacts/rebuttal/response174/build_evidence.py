"""Join existing policy-matched evidence and draw the six predeclared workloads."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics

os.environ.setdefault('MPLCONFIGDIR','/tmp/daso_rebuttal_matplotlib174')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
SOURCES={}

def track(relative):
    path=BASE/relative
    SOURCES[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read(relative):
    return json.loads(track(relative).read_text())

def rows(relative):
    return list(csv.DictReader(track(relative).open()))

def save(name, values):
    with (OUT/name).open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)

budget=[]
counts=[]
for model,rid in [('Qwen3-8B','044'),('Llama-3.1-8B','045'),('Llama-2-7B','046')]:
    design=read(f'runs/{rid}/configs/design.json')
    quality=rows(f'runs/{rid}/results/quality_budget_audit.csv')
    counts.append({'model':model,'run':rid,'budget_labels':len(quality),
        'unique_selected_model_policies':len({r['policy'] for r in quality}),
        'mean_violations':sum(r['mean_budget_violation']=='True' for r in quality),
        'confirmation_guard_violations':sum(r['guard_budget_violation']=='True' for r in quality)})
    for r in quality:
        if int(r['validation_size'])!=64 or float(r['epsilon']) not in [.02,.04]:
            continue
        origin=design['policies'][r['policy']]['origins'][0]
        source=origin['run'];pid=origin['policy']
        source_policy=read(f'runs/{source}/configs/policies/{pid}.json')
        fresh_policy=read(f'runs/{rid}/configs/policies/{r["policy"]}.json')
        signature=hashlib.sha256(json.dumps(source_policy['method_map'],sort_keys=True).encode()).hexdigest()
        assert source_policy['method_map']==fresh_policy['method_map']
        assert signature==design['policies'][r['policy']]['map_sha256']
        assert SOURCES[f'runs/{source}/configs/policies/{pid}.json']==design['policies'][r['policy']]['policy_sha256']
        if source=='032':
            speed=next(x for x in rows(f'runs/{source}/results/formal_speed.csv') if x['policy']==pid)
            speedup=float(speed['speedup_bf16'])
        else:
            speed=next(x for x in rows(f'runs/{source}/results/sweep_all_results.csv') if x['policy']==pid)
            speedup=float(speed['e2e_ms_vs_bf16'])
        runtime=next(x for x in rows('responses/reviewer3/new_point_overheads_night161/snapshots/20261027_184651_final/runtime_points.csv') if x['run']==source and x['policy']==pid)
        mean=float(r['confirmation_delta_nll']);two_se=2*float(r['confirmation_paired_se'])
        assert mean+two_se<float(r['epsilon'])
        budget.append({'model':model,'epsilon':float(r['epsilon']),'validation_size':64,'policy':r['policy'],
            'map_sha256':signature,'confirmation_run':rid,'confirmation_delta_nll':mean,'confirmation_two_se':two_se,
            'confirmation_mean_plus_two_se':mean+two_se,'validation_guard':float(r['validation_guard']),
            'confirmation_guard':float(r['confirmation_guard']),'mean_budget_violation':r['mean_budget_violation'],
            'guard_budget_violation':r['guard_budget_violation'],'speed_source_run':source,'speed_source_policy':pid,
            'inherited_e2e_speedup_bf16':speedup,'inherited_e2e_ms':float(speed['e2e_ms']),
            'inherited_cycle_speedup_int8':float(runtime['cycle_vs_int8']),
            'scope':'Exact policy/map match; NLL newly confirmed, speed inherited from origin campaign, not new simultaneous evaluation.'})
save('budget_speed.csv',budget)
save('budget_counts.csv',counts)

gate=rows('night161/latency_validation/latency_validation.csv')
seal=read('night161/latency_validation/preregistered.json')
gate_audit=read('night161/latency_validation/final_audit.json')
assert gate_audit['verified'] and len(gate)==36
assert SOURCES['night161/latency_validation/preregistered.json']==gate_audit['preregistered_sha256']
code=track('night161/code/latency_gate_analysis.py')
assert hashlib.sha256(code.read_bytes()).hexdigest()==seal['code_sha256']
gate_summary=[]
for scene,mode in [('prefill_only','same_operator'),('prefill_decode','same_operator'),('prefill_decode','phase_pair')]:
    for metric in ['e2e_ms','cycle_ms']:
        group=[r for r in gate if (r['scenario'],r['phase_mode'],r['metric'])==(scene,mode,metric)]
        assert len(group)==6
        gate_summary.append({'scenario':scene,'phase_mode':mode,'metric':metric,'labels':len(group),
            'feasible_labels':sum(r['selection_feasible']=='True' for r in group),
            'unique_selected_maps':len({r['selected_policy'] for r in group if r['selection_feasible']=='True'}),
            'original_misses':sum(int(r['original_misses']) for r in group),
            'selected_misses':sum(int(r['selected_misses']) for r in group if r['selection_feasible']=='True'),
            'confirmation_observations_per_label':16})
save('latency_gate_summary.csv',gate_summary)

work=rows('runs/040/results/workloads.csv')
rounds=rows('runs/040/results/speed_rounds.csv')
workloads=['center','single','short','long','decode256','single256']
policies=['w1_e0.01','w32_e0.03']
plot_data=[]
for workload in workloads:
    for pid in policies:
        aggregate=next(r for r in work if (r['workload'],r['policy'])==(workload,pid))
        group=[r for r in rounds if (r['workload'],r['policy'])==(workload,pid)]
        assert len(group)==3
        for metric in ['e2e_ms_vs_int8','cycle_ms_vs_int8']:
            vals=[float(r[metric]) for r in group];central=float(aggregate[metric])
            assert math.isclose(central,statistics.median(vals),abs_tol=1e-12)
            plot_data.append({'workload':workload,'batch':aggregate['batch'],'prompt':aggregate['prompt'],
                'output':aggregate['output'],'policy':pid,'metric':metric,'median_ratio':central,
                'minimum_ratio':min(vals),'maximum_ratio':max(vals),'rounds':3})
save('workload_figure_data.csv',plot_data)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
x=np.arange(6);width=.32
colors=['#176A91','#D77A21']
for ax,metric,title in zip(axes,['e2e_ms_vs_int8','cycle_ms_vs_int8'],['End-to-end','Full cycle (includes restoration)']):
    for index,pid in enumerate(policies):
        values=[next(r for r in plot_data if (r['workload'],r['policy'],r['metric'])==(w,pid,metric)) for w in workloads]
        med=np.array([r['median_ratio'] for r in values])
        err=np.array([[r['median_ratio']-r['minimum_ratio'] for r in values],[r['maximum_ratio']-r['median_ratio'] for r in values]])
        ax.errorbar(x+(index-.5)*width,med,yerr=err,color=colors[index],
            marker=['o','s'][index],linestyle='none',markersize=4,
            label=['DASO D1','DASO D32'][index],zorder=3,elinewidth=.9,capsize=2,capthick=.8)
    ax.axhline(1,color='#555555',ls='--',lw=.8,zorder=4)
    ax.set_title(title,pad=8)
    ax.set_xticks(x,['8/2048\n/64','1/2048\n/128','8/512\n/128','8/4096\n/64','8/512\n/256','1/512\n/256'])
    ax.set_xlabel('Batch / input / output tokens')
    ax.grid(axis='y',lw=.5,alpha=.3,zorder=0)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Speed relative to INT8 (higher is faster)')
upper=max(r['maximum_ratio'] for r in plot_data)
axes[0].set_ylim(.95,math.ceil(upper*20)/20+.05)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.015))
fig.tight_layout(rect=(0,0,1,.94),w_pad=1.3)
for extension in ['pdf','svg','png']:
    fig.savefig(OUT/f'qwen_six_workloads.{extension}',dpi=220,bbox_inches='tight')
plt.close(fig)
readme='''# Six-workload figure

Fixed Qwen3 policies D1 (`w1_e0.01`) and D32 (`w32_e0.03`), with no per-workload retuning. All six predeclared run040 cases are shown. Markers are the medians of three per-round speed ratios to INT8; whiskers are the full three-round minimum/maximum, not confidence intervals. End-to-end includes the prefill-to-decode transition; full cycle additionally includes restoration for the next prefill. This is a comparison to INT8, not a claim of dominance over all baselines or budget satisfaction at every workload. Some cycle ranges include parity even where the median is above one.

[PDF](qwen_six_workloads.pdf) · [SVG](qwen_six_workloads.svg) · [PNG](qwen_six_workloads.png) · [exact plotted data](workload_figure_data.csv) · [source](../runs/040/REPORT.md)
'''
(OUT/'FIGURE.md').write_text(readme)
(OUT/'analysis_audit.json').write_text(json.dumps({'verified':True,'source_sha256':SOURCES,
    'matched_budget_speed_points':len(budget),'budget_label_count':sum(r['budget_labels'] for r in counts),
    'unique_selected_model_policies':sum(r['unique_selected_model_policies'] for r in counts),
    'mean_violations':sum(r['mean_violations'] for r in counts),
    'confirmation_guard_violations':sum(r['confirmation_guard_violations'] for r in counts),
    'latency_labels':len(gate),'latency_unique_maps':gate_audit['unique_selected_maps'],
    'original_latency_misses':sum(int(r['original_misses']) for r in gate),
    'selected_latency_misses':sum(int(r['selected_misses']) for r in gate),
    'scope':'CPU analysis of existing frozen data. Labels/metrics share maps and measurements; no independent-trial count or hard SLA guarantee. No new GPU experiment.'},indent=2)+'\n')
print(json.dumps({'verified':True,'budget_speed':budget,'budget_counts':counts,'latency_summary':gate_summary,'sources':len(SOURCES)},indent=2))
