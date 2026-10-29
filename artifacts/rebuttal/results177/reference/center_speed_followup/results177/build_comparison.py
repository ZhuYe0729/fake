"""Reproduce measured per-load quality and publish the full strong-baseline comparison."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import os
import statistics

os.environ.setdefault('MPLCONFIGDIR','/tmp/daso_rebuttal_matplotlib177')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'responses/reviewer2/qwen_workload_comparison'
ART=Path(__file__).resolve().parent
RUN=BASE/'runs/040'
REPO=BASE.parent.parent
SOURCES={}

def track(path):
 path=Path(path);SOURCES[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest();return path

def read(path):return json.loads(track(path).read_text())
def records(path):return list(csv.DictReader(track(path).open()))
def save(path,rows):
 with path.open('w') as handle:
  writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def f(value,n=3):return f'{float(value):.{n}f}'
def signature(mapping):return hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()

raw=records(RUN/'results/workloads.csv')
rounds=records(RUN/'results/speed_rounds.csv')
design=read(RUN/'configs/design.json')
parent=read(RUN/'results/final_audit.json')
assert parent['verified'] and parent['complete'] and len(raw)==54 and len(rounds)==162
for path,digest in parent['evidence_sha256'].items():assert hashlib.sha256(track(RUN/path).read_bytes()).hexdigest()==digest,path
main={r['method']:r for r in read(BASE/'responses/reviewer2/qwen3_main_table/provenance.json')['rows']}
methods=[('BF16','uniform_dense_bf16'),('INT8','uniform_int8'),('SmoothQuant INT8','uniform_sq_int8'),('FP8','uniform_fp8'),
 ('SQ-INT8 → W4A16','fixed_sq_int8_w4a16'),('HAWQ-V3 (Balanced)','hawq_signed_prefill_decode_phase_pair_r0.8'),
 ('HAWQ-V3 (Max-speed)','hardware_minimum_prefill_decode_phase_pair'),('DASO (Balanced)','w1_e0.01'),('DASO (Max-speed)','w32_e0.03')]
for label,pid in methods:
 mapping=read(RUN/f'configs/policies/{pid}.json')['method_map']
 assert signature(mapping)==design['policies'][pid]['map_sha256']
 if label in main:assert signature(mapping)==main[label]['prefill_decode']['map_sha256']
 # INT8/FP8 display abbreviations differ from the primary table; their origin signatures are fixed in design.
code_equal={}
for name in ['common.py','eightbit.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py']:
 a=track(RUN/'code'/name);b=track(BASE/'runs/048/code'/name)
 code_equal[name]=hashlib.sha256(a.read_bytes()).hexdigest()==hashlib.sha256(b.read_bytes()).hexdigest()
assert all(code_equal.values())

by={(r['workload'],r['policy']):r for r in raw}
quality={}
for case in design['workloads']:
 for label,pid in methods:
  d=read(RUN/f'results/quality/{pid}/{case["label"]}.json')
  assert d['complete'] and d['quality_batch']==case['batch'] and d['quality_sha256']==case['quality_sha256']
  assert len(d['blocks'])==64 and [r['block'] for r in d['blocks']]==list(range(64))
  assert all(r['token_count']==case['output'] for r in d['blocks'])
  quality[case['label'],pid]=[float(r['avg_nll']) for r in d['blocks']]

full=[];plot_rows=[]
for case in design['workloads']:
 w=case['label'];reference=quality[w,'uniform_dense_bf16']
 for label,pid in methods:
  r=by[w,pid]
  differences=[a-b for a,b in zip(quality[w,pid],reference)]
  mean=statistics.mean(differences);se=statistics.stdev(differences)/8
  assert math.isclose(mean,float(r['delta_nll']),abs_tol=1e-10)
  assert math.isclose(se,float(r['paired_se']),abs_tol=1e-10)
  rs=[x for x in rounds if x['workload']==w and x['policy']==pid];assert len(rs)==3
  for key in ['e2e_ms','ttft_ms','tpot_ms','restore_ms','cycle_ms','e2e_ms_vs_bf16','ttft_ms_vs_bf16','tpot_ms_vs_bf16','cycle_ms_vs_bf16']:
   assert math.isclose(float(r[key]),statistics.median(float(x[key]) for x in rs),rel_tol=1e-10,abs_tol=1e-10),(w,pid,key)
  values=[float(x['e2e_ms_vs_bf16']) for x in rs]
  entry={'workload':w,'batch':case['batch'],'prompt':case['prompt'],'output':case['output'],'method':label,
   'source_policy':pid,'delta_nll':mean,'paired_se':se,'two_se':2*se,'quality_windows':64,
   'ttft_speedup_bf16':float(r['ttft_ms_vs_bf16']),'tpot_speedup_bf16':float(r['tpot_ms_vs_bf16']),
   'e2e_speedup_bf16':float(r['e2e_ms_vs_bf16']),'cycle_speedup_bf16':float(r['cycle_ms_vs_bf16']),
   'ttft_ms':float(r['ttft_ms']),'tpot_ms':float(r['tpot_ms']),'e2e_ms':float(r['e2e_ms']),
   'restore_ms':float(r['restore_ms']),'cycle_ms':float(r['cycle_ms']),
   'e2e_speedup_min':min(values),'e2e_speedup_max':max(values)}
  full.append(entry);plot_rows.append(entry)
save(OUT/'full_comparison.csv',full)
lookup={(r['workload'],r['method']):r for r in full}
selected={'SQ-INT8 → W4A16','HAWQ-V3 (Balanced)','HAWQ-V3 (Max-speed)','DASO (Balanced)','DASO (Max-speed)'}
overview=[r for r in full if r['method'] in selected]
save(OUT/'overview_comparison.csv',overview)

summary=[]
for case in design['workloads']:
 w=case['label'];all_rows=[r for r in full if r['workload']==w]
 ha=lookup[w,'HAWQ-V3 (Max-speed)'];a=lookup[w,'DASO (Balanced)'];b=lookup[w,'DASO (Max-speed)']
 fastest=max(all_rows,key=lambda r:r['e2e_speedup_bf16'])
 summary.append({'workload':w,'batch':case['batch'],'prompt':case['prompt'],'output':case['output'],
  'balanced_vs_hawq_max_e2e':a['e2e_speedup_bf16']/ha['e2e_speedup_bf16'],
  'maxspeed_vs_hawq_max_e2e':b['e2e_speedup_bf16']/ha['e2e_speedup_bf16'],
  'balanced_minus_hawq_max_nll':a['delta_nll']-ha['delta_nll'],
  'maxspeed_minus_hawq_max_nll':b['delta_nll']-ha['delta_nll'],
  'fastest_observed_e2e_method':fastest['method'],'fastest_observed_e2e_speedup':fastest['e2e_speedup_bf16']})
save(OUT/'summary.csv',summary)

headers=['方法','实测ΔNLL ±2SE ↓','P+D TTFT ×','P+D TPOT ×','P+D E2E ×','完整cycle ×']
def markdown(rows):
 speed_keys=['ttft_speedup_bf16','tpot_speedup_bf16','e2e_speedup_bf16','cycle_speedup_bf16']
 best={k:max(float(f(r[k])) for r in rows) for k in speed_keys};qbest=min(float(f(r['delta_nll'],5)) for r in rows)
 lines=['| '+' | '.join(headers)+' |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for r in rows:
  label='**'+r['method']+'**' if r['method'].startswith('DASO') else r['method']
  q=f(r['delta_nll'],5)+' ± '+f(r['two_se'],5)
  if float(f(r['delta_nll'],5))==qbest:q='**'+q+'**'
  values=[]
  for k in speed_keys:
   val=f(r[k]);values.append('**'+val+'**' if float(val)==best[k] else val)
  lines.append('| '+' | '.join([label,q,*values])+' |')
 return '\n'.join(lines)

lines=['# Qwen六种P+D负载：强基线的质量与速度完整对照','',
 '**精度列为按各B/S/O实际运行测得的全模型ΔNLL，不是预测值，也不是ROUGE/BLEU的Avg。** 每种负载包含BF16、INT8、SQ-INT8、FP8、固定阶段组合，以及HAWQ/DASO各两个工作点，共54行。','',
 'ΔNLL越低越好；负数表示这些窗口上的NLL低于BF16。全部速度相对同负载BF16，越高越快；cycle额外包含下一批Prefill前恢复。所有配置固定迁移，没有逐负载调参。','',
 '**读表结论：** DASO Balanced的E2E中位数在6/6负载快于HAWQ Max-speed，DASO Max-speed在5/6负载快于它；HAWQ通常有更低的NLL损失。全部九方法中，两个DASO点组成的速度上包络在4/6负载最快，另两种长Decode负载是固定SQ-INT8→W4A16更快。保留这些取舍，不宣称全面支配。','',
 '[分组HTML](table.html) · [全部54行CSV](full_comparison.csv) · [直接强对照30行](overview_comparison.csv) · [质量–速度PDF图](quality_speed.pdf) · [SVG](quality_speed.svg)','']
for case in design['workloads']:
 w=case['label'];lines+=['## B='+str(case['batch'])+' / 输入='+str(case['prompt'])+' / 输出='+str(case['output']),'',markdown([r for r in full if r['workload']==w]),'']
notes='''- 同GPU6/NUMA1、KV6GiB、共同native decode Graph；每个策略每轮初始化一次、顺序测六个负载，三轮每case 3 warmup/5 timed。表中为三轮中位数，各速度有本轮BF16首尾锚点。
- 精度来自实际目标batch下64个WikiText窗口的逐token teacher forcing，使用真实KV路径并核对目标token；同负载逐窗口减BF16后计算均值与SE。窗口此前已研究，跨负载有共享，不是新外部盲测。
- 质量数据是对应B/S/O的实测值；没有把中心模型主表的生成Avg复制给所有负载。三项下游任务没有按这六个固定负载分别重跑，因此本表不提供该Avg。
- HAWQ为当前FP4/INT8/BF16后端的Hessian/ILP PTQ适配；Max-speed是已验证的最小可行延迟端点，与模型主表使用相同map。DASO两个点也与模型主表的P+D配置一致；名称不表示在每个新负载都速度最高。
- 本表统一使用040的质量与速度。模型主表048是另一轮测量，数值不混拼；数值代码相同，测量轮次/负载顺序不同，变化来源未经因果确认。不能把主表更有利的中心速度替入这里。
- ±2SE是窗口经验统计，图上横线是三轮最小/最大范围，不是性能置信区间。实测均值、带余量保护值和下游任务分数是不同指标，不能用原代理q充当跨负载实际质量保证。
- 数值粗体为本负载展示精度下的最高速度或最低平均NLL；仅为点估计比较，不作显著性声明。完整绝对TTFT/TPOT/E2E、恢复及每轮范围见CSV与来源。
'''
lines+=['<details>','<summary>测量与精度范围说明</summary>','',notes,'[原始实验、代码和完整审计](../../../runs/040/README.md) · [本次来源与重算](provenance.json)','','</details>','']
(OUT/'README.md').write_text('\n'.join(lines))

html_parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen多负载完整对照</title><style>body{font:15px system-ui;max-width:1150px;margin:30px auto;padding:0 16px;color:#172330}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #dfe5e9;white-space:nowrap}th:first-child{text-align:left}thead{background:#edf2f6}.daso{background:#eaf5fc}.hawq{background:#fff9ed}.scroll{overflow:auto}p,li{line-height:1.65}h2{margin-top:30px}</style><h1>Qwen六负载：精度与速度完整对照</h1><p>精度为各负载实测ΔNLL（越低越好），不是ROUGE/BLEU Avg。速度相对BF16（越高越好）。同一轮共同测量，全部九方法均保留。</p>']
for case in design['workloads']:
 html_parts.append(f'<h2>B={case["batch"]} / 输入={case["prompt"]} / 输出={case["output"]}</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>')
 for r in [r for r in full if r['workload']==case['label']]:
  cls='daso' if r['method'].startswith('DASO') else 'hawq' if r['method'].startswith('HAWQ') else ''
  vals=[r['method'],f(r['delta_nll'],5)+' ± '+f(r['two_se'],5),*[f(r[k]) for k in ['ttft_speedup_bf16','tpot_speedup_bf16','e2e_speedup_bf16','cycle_speedup_bf16']]]
  html_parts.append('<tr class="'+cls+'">'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
 html_parts.append('</tbody></table></div>')
html_parts+=['<p><a href="full_comparison.csv">完整CSV</a> · <a href="README.md">Markdown和实验说明</a> · <a href="quality_speed.pdf">PDF图</a></p>','</html>']
(OUT/'table.html').write_text(''.join(html_parts))

# All nine measured configurations are plotted; repeat ranges and NLL uncertainty remain visible.
styles={
 'BF16':('#555555','x'),'INT8':('#999999','^'),'SmoothQuant INT8':('#5b8c5a','v'),'FP8':('#aa6677','P'),
 'SQ-INT8 → W4A16':('#8357a0','D'),'HAWQ-V3 (Balanced)':('#d4a43b','o'),'HAWQ-V3 (Max-speed)':('#b56917','^'),
 'DASO (Balanced)':('#1874a6','o'),'DASO (Max-speed)':('#06496e','s')}
plot_labels={**{label:label for label,_ in methods},'SQ-INT8 → W4A16':'SQ-INT8 / W4A16'}
plt.rcParams.update({'font.size':7,'axes.titlesize':8,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(8.4,5.2),sharex=True,sharey=True)
for ax,case in zip(axes.flat,design['workloads']):
 for label,pid in methods:
  r=lookup[case['label'],label];color,marker=styles[label]
  ax.errorbar(r['e2e_speedup_bf16'],r['delta_nll'],
   xerr=[[r['e2e_speedup_bf16']-r['e2e_speedup_min']],[r['e2e_speedup_max']-r['e2e_speedup_bf16']]],
   yerr=r['two_se'],color=color,marker=marker,markersize=4,linestyle='none',
   elinewidth=.65,capsize=1.6,alpha=.95,label=plot_labels[label])
 ax.axhline(0,color='#888888',lw=.6,ls='--');ax.grid(alpha=.18,linewidth=.5)
 ax.set_title(f'B{case["batch"]} / S{case["prompt"]} / O{case["output"]}')
 ax.spines[['top','right']].set_visible(False)
for ax in axes[-1]:ax.set_xlabel('E2E speedup vs BF16 (higher better)')
for ax in axes[:,0]:ax.set_ylabel('Measured delta NLL (lower better)')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01),columnspacing=1.3)
fig.tight_layout(rect=(0,.15,1,1),h_pad=1.3,w_pad=1.2)
for extension in ['pdf','svg','png']:fig.savefig(OUT/f'quality_speed.{extension}',dpi=220,bbox_inches='tight')
plt.close(fig)

facts={'balanced_faster_than_hawq_max_cases':sum(r['balanced_vs_hawq_max_e2e']>1 for r in summary),
 'maxspeed_faster_than_hawq_max_cases':sum(r['maxspeed_vs_hawq_max_e2e']>1 for r in summary),
 'daso_envelope_fastest_cases':sum(r['fastest_observed_e2e_method'].startswith('DASO') for r in summary),
 'simple_phase_fastest_cases':sum(r['fastest_observed_e2e_method']=='SQ-INT8 → W4A16' for r in summary)}
assert facts=={'balanced_faster_than_hawq_max_cases':6,'maxspeed_faster_than_hawq_max_cases':5,'daso_envelope_fastest_cases':4,'simple_phase_fastest_cases':2}
provenance={'verified':True,'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':SOURCES,
 'methods':[{'display_method':label,'policy':pid,'map_sha256':design['policies'][pid]['map_sha256']} for label,pid in methods],
 'workloads':design['workloads'],'rows':54,'overview_rows':30,'windows_per_method_case':64,'quality_computed_from_raw_blocks':True,
 'latency_medians_recomputed':True,'source_evidence_hashes_verified':len(parent['evidence_sha256']),
 'numerical_code_equal_to_primary_campaign':code_equal,'facts':facts,
 'quality_scope':'Actual per-workload paired teacher-forced delta NLL. Not downstream task Avg, new blind data, or guaranteed feasibility under original proxy labels.',
 'new_gpu_measurements':0,'source_result_mutations':0}
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
(ART/'comparison_facts.json').write_text(json.dumps({'verified':True,'facts':facts,'quality_metric':'measured delta NLL','downstream_avg_repeated_for_workloads':False},indent=2)+'\n')
print(json.dumps({'verified':True,'rows':54,'overview_rows':30,'source_files':len(SOURCES),'facts':facts},ensure_ascii=False,indent=2))
