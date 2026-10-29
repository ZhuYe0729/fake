"""Report the complete frozen run005 P+D budget grid; no new measurements."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/005'
OUT=BASE/'responses/reviewer1/pd_quality_budget';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
def tracked(p):sources[str(p.resolve())]=sha(p);return p
design=read(tracked(RUN/'configs/design.json'));pool=read(tracked(RUN/'configs/candidate_pool.json'))
source_rows=list(csv.DictReader(tracked(RUN/'results/summary_final.csv').open()))
baseline=read(tracked(RUN/'results/test/uniform_dense_bf16/prefill_decode.json'))
assert baseline['data_sha256']==design['data']['test']['file_sha256'] and len(baseline['blocks'])==64
rows=[]
for family in ['original DP','exact forward']:
 selected=sorted([r for r in source_rows if r['scenario']=='prefill_decode' and r['family']==family],key=lambda r:float(r['quality_budget']))
 assert [float(r['quality_budget']) for r in selected]==design['quality_budgets']
 for row in selected:
  policy=row['policy_id'];point=next(p for p in pool['points'] if p['label']==row['label'])
  assert point['family']==family and point['predicted_delta_nll']==float(row['predicted_delta_nll'])
  assert point['predicted_delta_nll']<=point['quality_budget']+1e-12
  raw=read(tracked(RUN/f'results/test/{policy}/prefill_decode.json'))
  assert raw['mode']=='test' and raw['quality_score_batch']==8 and len(raw['blocks'])==64
  assert raw['data_sha256']==baseline['data_sha256']
  for a,b in zip(raw['blocks'],baseline['blocks'],strict=True):assert a['block']==b['block'] and a['token_count']==b['token_count']==64
  delta=[a['avg_nll']-b['avg_nll'] for a,b in zip(raw['blocks'],baseline['blocks'],strict=True)]
  mean=statistics.mean(delta);se=statistics.stdev(delta)/8
  assert math.isclose(mean,float(row['measured_delta_nll']),rel_tol=0,abs_tol=1e-12)
  rows.append({'family':family,'budget':point['quality_budget'],'predicted_delta_nll':point['predicted_delta_nll'],
   'measured_mean_delta_nll':mean,'test_windows':64,'target_tokens_per_window':64,'paired_se':se,
   'measured_minus_predicted':mean-point['predicted_delta_nll'],'mean_budget_violation':max(0.,mean-point['quality_budget']),
   'mean_within_budget':mean<=point['quality_budget'],'bootstrap_ci_low':float(row['delta_nll_ci_low']),
   'bootstrap_ci_high':float(row['delta_nll_ci_high']),'policy_id':policy,'source_run':'005'})
with (OUT/'points.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def table(family):
 return ['| 预算 ε | 预测平均 ΔNLL | 实测平均 ΔNLL |','| ---: | ---: | ---: |',
  *[f"| {r['budget']:.3f} | {r['predicted_delta_nll']:.5f} | {r['measured_mean_delta_nll']:.5f} |" for r in rows if r['family']==family]]
primary=[r for r in rows if r['family']=='original DP']
errors=[abs(r['measured_minus_predicted']) for r in primary]
qwen_metrics=read(tracked(BASE/'runs/002/results/daso8/prefill_decode/quality_metrics.json'))
paper_metrics=read(tracked(BASE.parents[1]/'daso/results/paper/modeling_calibration/llama31_8b_instruct/prefill_decode/quality/metrics.json'))
context={'unit':'nats/token; absolute delta NLL, not percent','primary_four_policy_mae':statistics.mean(errors),
 'primary_max_absolute_error':max(errors),'qwen_holdout_mae':qwen_metrics['holdout']['mae'],
 'paper_llama31_holdout_mae':paper_metrics['metrics']['holdout']['mae'],
 'scope':'Different models, policy sets and text samples; magnitude context only, not a matched improvement comparison.'}
(OUT/'error_context.json').write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n')
lines=['# 历史P+D质量预算诊断：预测与实测（R1 / AK6a）','',
 '状态：内部诊断，暂不作为最终rebuttal主表。此处保留早期Run005原DP及精确组合求解的完整四预算，用于核对质量模型偏差与求解版本。','',
 '模型为Qwen3-8B，B=8、输入2048、输出64；每个策略评测同一组64个独立于拟合的WikiText测试窗口，每窗64个teacher-forced目标token，共4096个计分token。单位均为nats/token。',
 '主表采用原DP、Run005当时冻结的七种候选和质量拟合，直接审计求解输出，未经过新增validation筛选；它不是最新主表代表点的重新评测，也不代表全部模型或Prefill场景。','',
 *table('original DP'),'','四个预测值均在预算内，四个测试平均值也均在预算内，观察到的平均超限数为0/4。负值表示这批窗口上的NLL略低于BF16。',
 f"四点绝对误差MAE为{context['primary_four_policy_mae']:.5f}，最大误差{context['primary_max_absolute_error']:.5f}；对应Qwen留出策略MAE为{context['qwen_holdout_mae']:.5f}，论文Llama-3.1留出MAE为{context['paper_llama31_holdout_mae']:.5f}。这些均是绝对ΔNLL单位，不是百分比；集合不同，仅说明误差尺度，不证明相同条件下预测改善。接近零的预测/实测比值会显得很大，小预算的约束可靠性仍需单独检查。",'',
 '较宽预算下预测偏保守，不能将本表描述为预测与实测始终接近。最紧预算0.005的实测均值虽未超限，其既有描述性95%配对bootstrap区间为[-0.02225, 0.00744]，跨过预算；不能据此声称严格质量保证或真实违约概率为零。','',
 '[逐点数值、SE与区间](points.csv) · [误差尺度](error_context.json) · [来源与重算审计](audit.json) · [原始完整报告](../overnight_run005/REPORT.md)','',
 '<details>','<summary>同一实验的精确求解版（独立版本，不能与原DP混合统计）</summary>','',
 '保持同一冻结质量代理和测试协议，仅更换为连续质量成本的精确组合求解；未经过validation筛选。两个版本分别统计，完整四预算结果保留。','',
 *table('exact forward'),'','预算0.005处，预测0.00470348、实测0.00574726，平均超限0.00074726；其余三点均值未超限。较宽预算下依然存在明显保守预测，不能以“高精度预测”概括。','',
 '</details>','',
 '本表整理已冻结实验，策略、预算、数据和预测值均沿用原记录，未新增测量。有限预算网格上的通过率不等同于每条请求的质量保证。','']
(OUT/'README.md').write_text('\n'.join(lines))
audit={'verified':True,'new_measurements':False,'primary_family':'original DP','scope':'Qwen3-8B P+D B8/S2048/O64; run005 frozen seven-candidate setting',
 'primary_points':4,'primary_mean_violations':sum(not r['mean_within_budget'] for r in rows if r['family']=='original DP'),
 'appendix_exact_points':4,'appendix_exact_mean_violations':sum(not r['mean_within_budget'] for r in rows if r['family']=='exact forward'),
 'window_pairs_recomputed':len(rows)*64,'source_sha256':sources,'builder_sha256':sha(Path(__file__)),'points_sha256':sha(OUT/'points.csv')}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},ensure_ascii=False))
