from pathlib import Path
import json,csv,re
base=Path('artifacts/rebuttal');p=base/'EXPERIMENT_CHECKLIST.md';text=p.read_text()
# Five-budget derived study, with all unmeasured raw-budget entries kept explicit.
points=list(csv.DictReader((base/'runs/021/results/budget_points.csv').open()))
block=re.search(r'(## 表5 — .*?)(?=\n## 表6)',text,re.S);lines=block[1].splitlines()
for i,line in enumerate(lines):
 if not line.startswith('|'):continue
 cells=[v.strip() for v in line.split('|')[1:-1]]
 if len(cells)!=8:continue
 cells.insert(7,'ε=.07' if cells[0]=='模型' else '---' if cells[0]=='---' else '—')
 if cells[0] in ['Llama-2','Llama-3.1'] and cells[2]=='DASO-8 + validation':
  rid='018' if cells[0]=='Llama-2' else '017';scene='prefill_only' if cells[1]=='Prefill' else 'prefill_decode'
  for index,budget in enumerate([.005,.01,.02,.04,.07],start=3):
   row=next(x for x in points if x['source_run']==rid and x['scenario']==scene and float(x['quality_budget'])==budget)
   cells[index]=f'{float(row["guarded_delta_nll"]):.4f} / {float(row["confirmation_delta_nll"]):.4f}±{2*float(row["confirmation_se"]):.4f} / '+('超' if row['mean_budget_violation']=='True' else '通过')
  cells[-1]='021+022已完成；共用确认样本与重复map明确'
 lines[i]='| '+' | '.join(cells)+' |'
note='新增Llama validation行按“validation guard / test均值±2SE / 是否超限”填写；其他原始代理行仍使用其代理预测，未测预算为—。20点的正式速度与选中map见[完整五预算表](responses/reviewer1/five_budget_llama_night151/README.md)。'
lines.insert(2,note+'\n');text=text[:block.start()]+'\n'.join(lines)+'\n'+text[block.end():]
# Actual word-perplexity pruning comparison.
block=re.search(r'(## 表13 — .*?)(?=\n## 表14)',text,re.S);lines=block[1].splitlines()
pruning={r['method']:r for r in csv.DictReader((base/'runs/020/results/pruning_table.csv').open())}
llama2={r['method']:r for r in csv.DictReader((base/'runs/018/results/main_table.csv').open())}
for i,line in enumerate(lines):
 if not line.startswith('|'):continue
 cells=[v.strip() for v in line.split('|')[1:-1]]
 if cells[0]=='模型':cells[4]='WikiText word PPL';lines[i]='| '+' | '.join(cells)+' |';continue
 if cells[0]=='Llama-3.1':
  key={'BF16':'BF16','SparseGPT 2:4':'SparseGPT 2:4','Wanda 2:4':'Wanda 2:4','Magnitude 2:4':'Magnitude 2:4'}[cells[1]]
  row=pruning[key];cells[2]='128' if key in ['SparseGPT 2:4','Wanda 2:4'] else '0' if key=='Magnitude 2:4' else '—'
  cells[3]=f'{float(row["mmlu"]):.3f}';cells[4]=f'{float(row["wikitext_word_ppl"]):.3f}'
  cells[5]=f'{float(row["prefill_ttft_speedup"]):.3f}';cells[6]=f'{float(row["pd_e2e_speedup"]):.3f}'
  cells[7]='历史缓存；原计时未恢复' if key=='SparseGPT 2:4' else '—' if key=='BF16' else f'{json.loads((base/"runs/020/results"/f"canonical_{key.split()[0].lower()}.json").read_text())["total_seconds_including_save"]:.2f}'
  cells[8]='017/020完整质量与三轮速度已完成'
 elif cells[0]=='Llama-2' and cells[1] in ['BF16','SparseGPT 2:4']:
  method='BF16' if cells[1]=='BF16' else 'Sparse BF16 (SparseGPT)';row=llama2[method]
  pid='uniform_dense_bf16' if cells[1]=='BF16' else 'uniform_sparse_bf16'
  ppl=json.loads((base/f'runs/018/results/tasks/{pid}/wikitext/full/result.json').read_text())['metrics']['word_perplexity,none']
  cells[3]=f'{float(row["mmlu"]):.3f}';cells[4]=f'{ppl:.3f}';cells[5]=f'{float(row["prefill_ttft_speedup"]):.3f}';cells[6]=f'{float(row["pd_e2e_speedup"]):.3f}';cells[8]='018统一新路径已完成；其他剪枝仍待'
 if cells[0] in ['Llama-3.1','Llama-2']:lines[i]='| '+' | '.join(cells)+' |'
lines.insert(2,'Llama-3.1的四MC、完整word PPL及三生成/三轮速度已齐，见[完整剪枝表](responses/reviewer4/pruning_llama31_run020/README.md)。SG在MMLU/PPL领先，Wanda在部分生成分数更高；不将任何方法称为所有指标最优。\n')
text=text[:block.start()]+'\n'.join(lines)+'\n'+text[block.end():]
text=text.replace('当前019固定11标签/10map、六负载及每负载64窗口质量与三轮速度已排队；包含O256，新DASO-8和强硬件端点，等第一份原模型核心表完成后复用空闲卡。见[runs019](runs/019/README.md)。','019固定11标签/10map、六负载及每负载64窗口质量/三轮速度已完成；含O256，不按新负载重选。DASO-8 ε=.04的E2E数值最快5/6、cycle最快4/6，6个NLL均值均在预算内。见[完整结果](responses/reviewer2/workloads_run019/README.md)。')
p.write_text(text)
print('Five-budget and pruning tables populated with measured values')
