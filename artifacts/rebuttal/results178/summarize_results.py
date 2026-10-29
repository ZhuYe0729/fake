"""Present the completed core campaign against the strongest measured HAWQ point."""
from pathlib import Path
import csv
import json

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049'
audit=json.loads((OUT/'audit.json').read_text());assert audit['complete'] and audit['tier']==1
stability=json.loads((OUT/'stability.json').read_text());assert stability['complete']
rows=list(csv.DictReader((OUT/'comparison.csv').open()))
by={(r['workload'],r['method']):r for r in rows}
def f(value,n=3):return f'{float(value):.{n}f}'
def speed(row,key):return f(row[key])+('†' if row['timing_status']=='需复核' else '')
def table(headers,values):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in values])
pp=[];pd=[];facts={'prefill_balanced_quality_and_speed_better_cases':0,'pd_balanced_faster_than_hawq_speed_cases':0,'pd_speed_faster_than_hawq_speed_cases':0}
for w in ['center_pp','batch4_pp','prompt4096_pp']:
 h=by[w,'HAWQ-V3（均衡）'];d=by[w,'DASO（均衡）'];shared=by[w,'HAWQ-V3（速度优先）']
 pp.append([f"{h['batch']} / {h['prompt']}",f(h['delta_nll'],5),speed(h,'ttft_speedup'),f(d['delta_nll'],5),speed(d,'ttft_speedup'),f(shared['delta_nll'],5),speed(shared,'ttft_speedup')])
 facts['prefill_balanced_quality_and_speed_better_cases']+=float(d['delta_nll'])<float(h['delta_nll']) and float(d['ttft_speedup'])>float(h['ttft_speedup'])
ratios=[];cycles=[]
for w in ['center_pd','batch4_pd','prompt4096_pd','output128_pd']:
 h=by[w,'HAWQ-V3（速度优先）'];d=by[w,'DASO（均衡）'];s=by[w,'DASO（速度优先）']
 pd.append([f"{h['batch']} / {h['prompt']} / {h['output']}",f(h['delta_nll'],5),speed(h,'e2e_speedup'),f(d['delta_nll'],5),speed(d,'e2e_speedup'),f(s['delta_nll'],5),speed(s,'e2e_speedup')])
 facts['pd_balanced_faster_than_hawq_speed_cases']+=float(d['e2e_speedup'])>float(h['e2e_speedup'])
 facts['pd_speed_faster_than_hawq_speed_cases']+=float(s['e2e_speedup'])>float(h['e2e_speedup'])
 ratios.append(float(s['e2e_speedup'])/float(h['e2e_speedup']))
 cycles.append(float(s['cycle_speedup'])/float(h['cycle_speedup']))
lines=['# Qwen核心配置：双场景速读表（R2 / Exi5）','',
 '第一批四组配置、三种Prefill及四种P+D场景已完成；DASO/HAWQ各两个冻结工作点，BF16作参照。精度为每场景64窗实测ΔNLL，越低越好；速度为三轮中位数，相对BF16，越高越快。下游任务Avg未重测。','',
 '†表示该行速度轮间范围或BF16首尾漂移超过预设5%门槛，需复核，不能据中位数宣称稳定胜出。全部轮次保留，未按有利程度筛除。','',
 '## Prefill-only：均衡点与共同速度端点','',
 '两个方法的速度优先端点是相同配置，合并展示。这里保留HAWQ均衡点、DASO均衡点和共同端点，未按负载换点。','',
 table(['B / 输入','HAWQ均衡 ΔNLL','HAWQ均衡 TTFT×','DASO均衡 ΔNLL','DASO均衡 TTFT×','共同端点 ΔNLL','共同端点 TTFT×'],pp),'',
 '## Prefill + Decode：直接比较最强HAWQ速度点','',
 '完整表仍保留HAWQ均衡点；本摘要直接列出其速度优先端点和DASO两个点。','',
 table(['B / 输入 / 输出','HAWQ速度点 ΔNLL','HAWQ速度点 E2E×','DASO均衡 ΔNLL','DASO均衡 E2E×','DASO速度点 ΔNLL','DASO速度点 E2E×'],pd),'',
 f"Prefill均衡点在{facts['prefill_balanced_quality_and_speed_better_cases']}/3配置同时有更低的NLL和更快的TTFT。P+D中，DASO均衡/速度点分别在{facts['pd_balanced_faster_than_hawq_speed_cases']}/4、{facts['pd_speed_faster_than_hawq_speed_cases']}/4配置的E2E中位数快于HAWQ速度点；HAWQ通常NLL更低，属于质量–速度取舍。",'',
 f"逐轮配对看，{stability['cases_with_one_fixed_daso_point_faster_than_hawq_speed_in_all_three_rounds']}/4种P+D负载存在同一个冻结DASO点在三轮都快于HAWQ速度点。原场景与B4是DASO速度点，输出128是DASO均衡点；S4096差别很小且重复范围重叠，不作稳定领先结论。",'',
 f"DASO速度点相对HAWQ速度点的P+D E2E差异约为{(round(100*(min(ratios)-1),1) or 0.0):.1f}%–{100*(max(ratios)-1):.1f}%（S4096基本持平）；计入下一批恢复后的完整cycle相对差异为{100*(min(cycles)-1):+.1f}%–{100*(max(cycles)-1):+.1f}%。E2E与cycle分别报告，不将恢复开销省略。",'',
 '均衡/速度优先是固定配置名称，非逐负载最优声明；上述比较是点估计，完整±2SE、重复范围与稳定性标记在下方来源。共享公开WikiText窗口不是新的盲测，不作下游任务或硬预算保证。','',
 'Prefill均衡DASO是主表选出的SQ-INT8配置；HAWQ沿用主表FP4/INT8/BF16的PTQ适配。本轮检验整套系统工作点的负载迁移，不单独证明同候选空间下分配器的优劣。','',
 '[全部35行、TTFT/TPOT/E2E/cycle与协议](README.md) · [分组HTML](table.html) · [各轮速度](speed_rounds.csv) · [原始单次计时范围](sample_statistics.csv) · [进程与排序稳定性](stability.json) · [质量–速度PDF](quality_speed.pdf)','']
(OUT/'SUMMARY.md').write_text('\n'.join(lines))
facts.update(pd_speed_e2e_gain_range=[min(ratios)-1,max(ratios)-1],pd_speed_cycle_gain_range=[min(cycles)-1,max(cycles)-1],timing_rows_flagged=len(audit['unresolved_timing_rows']))
facts['pd_cases_with_same_fixed_point_faster_in_all_repeats']=stability['cases_with_one_fixed_daso_point_faster_than_hawq_speed_in_all_three_rounds']
(OUT/'summary.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(facts,ensure_ascii=False,indent=2))
