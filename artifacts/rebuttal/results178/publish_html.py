"""Render the current measured comparisons without choosing or dropping points."""
from pathlib import Path
import csv
import html
import json

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049'
rows=list(csv.DictReader((OUT/'comparison.csv').open()))
audit=json.loads((OUT/'audit.json').read_text())
def fmt(value,n=3):return '待测' if value=='' else f'{float(value):.{n}f}'
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen邻近负载双场景对照</title><style>body{font:15px system-ui;max-width:1280px;margin:28px auto;padding:0 18px;color:#172330}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #e0e6eb;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#eef3f7}.daso{background:#eaf5fc}.hawq{background:#fff9ed}.group td{border-top:2px solid #b7c5d1}.scroll{overflow:auto}p{line-height:1.65}.status{padding:10px;background:#f3f5f7}</style><h1>Qwen邻近负载：DASO与HAWQ-V3</h1>']
parts.append('<p class="status">'+('当前批次完成' if audit['complete'] else '运行中')+f"：NLL {audit['quality_rows']}/{len(rows)}行，三轮速度 {audit['three_round_speed_rows']}/{len(rows)}行。共享Prefill端点在两方法下各展示一次。"+'</p>')
parts.append('<p>精度为实测ΔNLL±2SE，越低越好；全部速度相对同场景BF16，越高越快。均衡/速度优先均为冻结配置，速度优先不保证每个负载最快。此处未测试下游任务Avg。</p>')
for scene,title in [('prefill_only','Prefill-only'),('prefill_decode','Prefill + Decode')]:
 columns=['B / 输入 / 输出','方法','实测ΔNLL ±2SE ↓','TTFT ×']+(['TPOT ×','E2E ×','完整cycle ×'] if scene=='prefill_decode' else [])+['测速状态']
 parts.append('<h2>'+title+'</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>')
 previous=None
 for r in rows:
  if r['scenario']!=scene:continue
  group=r['workload']!=previous;previous=r['workload']
  shape=f"{r['batch']} / {r['prompt']} / "+(r['output'] if scene=='prefill_decode' else '—')
  cells=[shape,r['method'],'待测' if r['delta_nll']=='' else fmt(r['delta_nll'],5)+' ± '+fmt(r['two_se'],5),fmt(r['ttft_speedup'])]
  if scene=='prefill_decode':cells.extend(fmt(r[k]) for k in ['tpot_speedup','e2e_speedup','cycle_speedup'])
  cells.append(r['timing_status'])
  cls=('daso' if r['method'].startswith('DASO') else 'hawq' if r['method'].startswith('HAWQ') else '')+(' group' if group else '')
  parts.append('<tr class="'+cls+'">'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
 parts.append('</tbody></table></div>')
round_link=' · <a href="speed_rounds.csv">各轮速度</a>' if (OUT/'speed_rounds.csv').exists() else ''
extra=' · <a href="SUMMARY.md">速读表</a>' if (OUT/'SUMMARY.md').exists() else ''
if (OUT/'quality_speed.pdf').exists():extra+=' · <a href="quality_speed.pdf">质量–速度PDF</a>'
parts.append('<p>三轮中位数，固定5次预热/10次正式计时。需复核标记为速度轮间范围或BF16首尾漂移超过5%，仍保留全部数据。±2SE是NLL统计，不是测速误差。<a href="README.md">完整协议与Markdown</a> · <a href="comparison.csv">全部CSV</a>'+round_link+extra+'</p></html>')
(OUT/'table.html').write_text(''.join(parts))
readme=OUT/'README.md'
text=readme.read_text()
links='[分组HTML](table.html)'
if (OUT/'SUMMARY.md').exists():links='[速读表](SUMMARY.md) · '+links
if (OUT/'quality_speed.pdf').exists():links+=' · [质量–速度PDF](quality_speed.pdf) · [SVG](quality_speed.svg)'
line=next(line for line in text.splitlines() if '[完整CSV](comparison.csv)' in line)
tail=line[line.index('[完整CSV](comparison.csv)'):]
readme.write_text(text.replace(line,links+' · '+tail))
print(json.dumps({'html_rows':len(rows),'nll_rows':audit['quality_rows'],'three_round_speed_rows':audit['three_round_speed_rows']}))
