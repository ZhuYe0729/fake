"""Publish only complete, independently verified long speed tables."""
from pathlib import Path
import csv
import hashlib
import json
import time

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'responses/reviewer2/qwen_night_cohorts_run053'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(OUT/'audit.json');check=read(OUT/'long_speed_delivery_audit.json')
assert audit['core_primary_gpu'] is not None and audit['long_primary_gpu'] is not None
assert check['verified'] and check['long_primary_gpu']==audit['long_primary_gpu']
rows=list(csv.DictReader((OUT/'long_primary.csv').open()))
assert len(rows)==10 and all(int(r['completed_rounds'])==3 for r in rows)
quality_out=BASE/'responses/reviewer2/qwen_long_nll_run054';quality=read(quality_out/'audit.json')
quality_complete=quality['main_complete']
if quality_complete:
 assert read(quality_out/'delivery_audit.json')['main_complete']
 nll={(v['policy'],v['workload']):v for v in csv.DictReader((quality_out/'comparison.csv').open())}
else:nll={}
status='输出256/512的长Decode速度也已各完成同卡三轮；长输出专门NLL'+('已补齐。' if quality_complete else '尚未补齐。')
lines=['<a id="long-decode"></a>','### 长Decode补充（R2 / R3）','',
 f"B=8、输入2048，输出256和512；全部五个冻结点均在GPU{audit['long_primary_gpu']}完成三轮，独立核验通过。",'',
 'Avg沿用原任务协议的同map/B8分数（生成上限256、允许EOS），不表示在本表固定长度重新执行了下游任务。'+
 (' 专门长输出64窗NLL见独立补测表，逐点GPU和共同BF参考卡已明示。' if quality_complete else ' 长输出专门NLL未完成；不以原Avg替代该测量。'),'',
 '[逐轮速度与范围](responses/reviewer2/qwen_night_cohorts_run053/speed_rounds.csv) · [三轮配对比较](responses/reviewer2/qwen_night_cohorts_run053/paired_comparisons.csv) · [长速度独立核验](responses/reviewer2/qwen_night_cohorts_run053/long_speed_delivery_audit.json)','']
for output in ['256','512']:
 lines += [f'**输出={output}**','',
 '| 方法 | 原任务 P+D Avg ↑ | TTFT× ↑ | TPOT× ↑ | E2E× ↑ | cycle× ↑ |'+(' ΔNLL ± 2SE |' if quality_complete else ''),
 '| --- | ---: | ---: | ---: | ---: | ---: |'+(' ---: |' if quality_complete else '')]
 for r in rows:
  if r['output']!=output:continue
  dagger='†' if float(r['relative_round_range'])>.05 or float(r['max_bf16_anchor_drift'])>.05 else ''
  line=f"| {r['method']} | {float(r['original_task_avg']):.2f} | {float(r['ttft_speedup']):.3f} | {float(r['tpot_speedup']):.3f} | {float(r['e2e_speedup']):.3f}{dagger} | {float(r['cycle_speedup']):.3f} |"
  if quality_complete:
   v=nll[(r['policy'],r['workload'])];line+=f" {float(v['delta_nll']):+.5f} ± {float(v['delta_2se']):.5f} |"
  lines.append(line)
 lines.append('')
lines+=['†沿用上方5%范围/锚点漂移标记；小幅E2E差异需结合逐轮配对与恢复开销阅读，不能据此宣称所有负载稳定最优。','']
if quality_complete:lines+=['[专门长输出NLL及逐窗来源](responses/reviewer2/qwen_long_nll_run054/README.md)','']
snapshot=BASE/'results181/reference'/f'final_publication_{time.time_ns()}';snapshot.mkdir()
selected=BASE/'REBUTTAL_SELECTED_EVIDENCE.md';index=BASE/'responses/README.md'
for p in [selected,index]:(snapshot/(p.name+'.before')).write_text(p.read_text())
text=selected.read_text();assert '<a id="long-decode"></a>' not in text
anchor='<details>\n<summary>旧暂估与原六负载历史结果（与053分开读取）</summary>'
assert text.count(anchor)==1;text=text.replace(anchor,'\n'.join(lines)+'\n'+anchor)
text=text.replace('长Decode 256/512正在继续。',status);selected.write_text(text)
text=index.read_text().replace('长Decode 256/512继续测试。',status)
text=text.replace('**当前正式测速：','**夜间测速交付：');index.write_text(text)
for p in [selected,index]:(snapshot/(p.name+'.after')).write_text(p.read_text())
record={'published_unix':time.time(),'long_csv_sha256':sha(OUT/'long_primary.csv'),'long_verification_sha256':sha(OUT/'long_speed_delivery_audit.json'),
 'quality_complete':quality_complete,'publisher_sha256':sha(Path(__file__)),'published_documents':{str(p):sha(p) for p in [selected,index]}}
(snapshot/'publication.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False))
