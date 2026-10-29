"""Publish readable result tables from the existing sealed experiment artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import re

BASE=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
REPO=BASE.parent.parent
SOURCES={}
TABLES={}

def track(path):
    path=Path(path)
    SOURCES[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read(relative):return json.loads(track(BASE/relative).read_text())
def records(relative):return list(csv.DictReader(track(BASE/relative).open()))
def save(path,values):
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
def f(value,n=2):return f'{float(value):.{n}f}'
def md(headers,values):
    numeric=[all(re.fullmatch(r'-?\d+(?:\.\d+)?%?',str(row[i]).strip('*')) for row in values) for i in range(len(headers))]
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---:' if n else '---' for n in numeric)+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in values])+'\n'
def sig(mapping):return hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
def policy(run,pid):return read(f'runs/{run}/configs/policies/{pid}.json')['method_map']

PP=['arc_c','arc_e','wino','mmlu','prefill_ttft_speedup']
PD=['cnn_dm','dialogsum','iwslt','pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']
COLUMNS=['model','method',*PP[:4],'prefill_avg_acc',PP[-1],*PD[:3],'pd_avg_score',*PD[3:]]
SHORT=[('method','方法'),('prefill_avg_acc','Prefill Avg ↑'),('prefill_ttft_speedup','Prefill TTFT ×'),
    ('pd_avg_score','P+D Avg ↑'),('pd_tpot_speedup','P+D TPOT ×'),('pd_e2e_speedup','P+D E2E ×')]
FULL=[('method','方法'),('arc_c','ARC-C'),('arc_e','ARC-E'),('wino','Wino'),('mmlu','MMLU'),
    ('prefill_avg_acc','Prefill Avg'),('prefill_ttft_speedup','Prefill TTFT ×'),('cnn_dm','CNN/DM'),
    ('dialogsum','DialogSum'),('iwslt','IWSLT'),('pd_avg_score','P+D Avg'),('pd_ttft_speedup','P+D TTFT ×'),
    ('pd_tpot_speedup','P+D TPOT ×'),('pd_e2e_speedup','P+D E2E ×')]


def quality(run,pid,scene,values):
    cells={}
    items=[('arc_challenge','acc_norm,none','arc_c',1172),('arc_easy','acc,none','arc_e',2376),
        ('winogrande','acc,none','wino',1267),('mmlu','acc,none','mmlu',14042)] if scene=='prefill_only' else [
        ('cnn_dm_1000','rougeL_percent','cnn_dm',1000),('dsum','rougeL_percent','dialogsum',1500),('IWSLT','sacre_bleu','iwslt',333)]
    for task,metric,key,count in items:
        path=f'runs/{run}/results/'+(f'tasks/{pid}/{task}/full/result.json' if scene=='prefill_only' else f'generation/{pid}/{task}/metrics_0000_end.json')
        raw=read(path)
        if scene=='prefill_only':
            assert raw['limit'] is None and raw['num_fewshot']==0
            actual=float(raw['metrics'][metric])*100
            counts=raw['raw_lm_eval']['n-samples']
            assert sum(r['effective'] for r in counts.values())==count
        else:
            actual=float(raw[metric]);assert raw['samples']==count and raw['max_new_tokens']==256
        assert math.isclose(actual,values[key],abs_tol=1e-10),(path,actual,values[key])
        cells[key]={'source':path,'metric':metric,'samples':count}
    return cells


model_specs=[('Llama-3.1-8B-Instruct','llama31_8b_instruct','llama31_main_table','017','034','038','042',
    'DASO-8 exploratory P+D, q=0.03, wP=1 (034)','w1_q0.03','DASO-8 exploratory P+D, q=0.04, wP=1 (034)','w1_q0.04'),
    ('Llama-2-7B-Chat','llama2_7b_chat','llama2_main_table','018','035','039','043',
    'DASO-8 exploratory P+D, q=0.03, wP=32 (035)','w32_q0.03','DASO-8 exploratory P+D, q=0.01, wP=1 (035)','w1_q0.01')]
model_deliveries=[]
for model,stem,dest,core,sweep,phase,sqfp,balanced,bpid,fast,fpid in model_specs:
    directory=BASE/'responses/reviewer2'/dest;directory.mkdir(exist_ok=True)
    source=f'responses/reviewer2/{stem}_table2_night161/main_table.csv'
    master={r['method']:r for r in records(source)}
    original={r['method']:r for r in read(f'responses/reviewer2/{stem}_table2/provenance.json')}
    allocation=read(f'runs/{core}/configs/allocation_pool.json')
    floor_proof={}
    for scene,mode in [('prefill_only','same_operator'),('prefill_decode','phase_pair')]:
        point=next(p for p in allocation['points'] if p['label']==f'hardware_minimum_{scene}_{mode}')
        matrix=read(f'runs/{core}/data/matched_matrix_{scene}_{mode}.json')
        actual=policy(core,point['policy_id'])
        for kind in ['endpoint_hawq_indices','endpoint_daso_indices']:
            mapping={name:{'prefill_method':matrix['pairs'][i][0],'decode_method':matrix['pairs'][i][1]}
                for name,i in zip(matrix['names'],point[kind])}
            assert mapping==actual
        minimum=sum(min(row) for row in matrix['costs'])
        assert math.isclose(minimum,point['minimum_linear_ms'],abs_tol=1e-7)
        floor_proof[scene]={'solver_point':point,'map_sha256':sig(actual),'minimum_linear_ms':minimum}
    def part(method,scene,run,pid,quality_run=None,quality_pid=None):
        fields=PP if scene=='prefill_only' else PD
        values={k:float(master[method][k]) for k in fields}
        qr=quality_run or run;qp=quality_pid or pid
        active=policy(run,pid);reference=policy(qr,qp)
        if scene=='prefill_only':assert {n:v['prefill_method'] for n,v in active.items()}=={n:v['prefill_method'] for n,v in reference.items()}
        else:assert active==reference
        return values,{'source_table':source,'source_row':method,'scenario':scene,'policy_run':run,'policy':pid,
            'quality_run':qr,'quality_policy':qp,'map_sha256':sig(active),'cells':quality(qr,qp,scene,values)}
    def old(method,scene):
        return part(method,scene,core,original[method]['policies'][scene])
    result=[];provenance=[]
    def add(name,pp,pd,definition=None):
        a,pa=pp;b,pb=pd;r={'model':model,'method':name,**a,**b}
        r['prefill_avg_acc']=sum(r[k] for k in PP[:4])/4;r['pd_avg_score']=sum(r[k] for k in PD[:3])/3
        result.append(r);provenance.append({'method':name,'prefill_only':pa,'prefill_decode':pb,'reporting_definition':definition})
    originals=[('BF16','BF16'),('Dense NVFP4','Dense NVFP4'),('W4A16 (FP4)','W4A16 NVFP4 Marlin'),
        ('Sparse BF16 (2:4)','Sparse BF16 (SparseGPT)'),('Sparse NVFP4 (2:4)','Sparse NVFP4 (SparseGPT)')]
    for display,method in originals:add(display,old(method,'prefill_only'),old(method,'prefill_decode'))
    for display,method,pid in [('INT8 (W8A8)','INT8 W8A8','uniform_int8'),('SmoothQuant INT8','SmoothQuant INT8','uniform_sq_int8'),('FP8 (W8A8)','FP8 W8A8','uniform_fp8')]:
        add(display,old(method,'prefill_only'),part(f'{method} ({sweep} reference)','prefill_decode',sweep,pid,core,pid))
    name=f'SmoothQuant FP8 alpha=.5 ({sqfp})'
    add('SmoothQuant FP8',part(name,'prefill_only',sqfp,'uniform_sq_fp8'),part(name,'prefill_decode',sqfp,'uniform_sq_fp8'))
    name='Uniform SQ-INT8 prefill / W4A16 decode';pid='phase_sq_int8_w4a16_ours'
    add('SQ-INT8 → W4A16',old('SmoothQuant INT8','prefill_only'),part(f'{name} ({sweep} reference)','prefill_decode',sweep,pid,core,pid))
    for display,base,label,pid in [('INT8 → W4A16','INT8 W8A8','Uniform INT8 prefill / W4A16 decode','fixed_int8_w4a16'),
        ('FP8 → W4A16','FP8 W8A8','Uniform FP8 prefill / W4A16 decode','fixed_fp8_w4a16')]:
        # Policy IDs are taken from the fixed control design, not inferred from the display name.
        read(f'runs/{phase}/configs/sweep_design.json')
        candidates=[p for p in (BASE/f'runs/{phase}/configs/policies').glob('*.json') if p.stem==pid]
        if not candidates:
            pre='int8' if display.startswith('INT8') else 'fp8'
            candidates=[p for p in (BASE/f'runs/{phase}/configs/policies').glob('*.json')
                if all(v['prefill_method']==pre and v['decode_method']=='w4a16_ours' for v in json.loads(p.read_text())['method_map'].values())]
        assert len(candidates)==1,candidates
        pid=candidates[0].stem
        add(display,old(base,'prefill_only'),part(f'{label} ({phase} fixed control)','prefill_decode',phase,pid))
    add('HAWQ-V3 (Balanced)',old('HAWQ-FP4 PTQ r=.80','prefill_only'),old('HAWQ-FP4 PTQ r=.80','prefill_decode'),'FP4 PTQ hardware adaptation, r=.80.')
    shared=old('Common hardware-minimum FP4/INT8 control','prefill_only')
    add('HAWQ-V3 (Max-speed)',shared,old('Common hardware-minimum FP4/INT8 control','prefill_decode'),floor_proof)
    pp_balanced=old('SmoothQuant INT8' if core=='017' else 'FP8 W8A8','prefill_only')
    add('DASO (Balanced)',pp_balanced,part(balanced,'prefill_decode',sweep,bpid),
        'Representative reported quality-speed point; independent scenario policies. P+D is an explored point, not a common actual epsilon guarantee.')
    add('DASO (Max-speed)',shared,part(fast,'prefill_decode',sweep,fpid),
        'Faster of the two displayed representative points, not the global fastest policy across all archived budgets. Prefill uses the verified common floor.')
    assert len(result)==16
    save(directory/'main_table.csv',[{key:r[key] for key in COLUMNS} for r in result])
    best={k:max(float(f(r[k])) for r in result) for k in COLUMNS[2:]}
    def table(fields):
        cells=[]
        for r in result:
            line=[]
            for key,_ in fields:
                val=r[key] if key=='method' else f(r[key])
                if (key=='method' and r[key].startswith('DASO')) or (key!='method' and float(val)==best[key]):val='**'+val+'**'
                line.append(val)
            cells.append(line)
        return md([title for _,title in fields],cells)
    by={r['method']:r for r in result}
    a=by['DASO (Balanced)'];h=by['HAWQ-V3 (Balanced)']
    conclusion=f"**结论：** P+D Balanced下，DASO为Avg {a['pd_avg_score']:.2f}、E2E {a['pd_e2e_speedup']:.2f}×；HAWQ对应为{h['pd_avg_score']:.2f}、{h['pd_e2e_speedup']:.2f}×。完整baseline均在表中，可直接比较质量–速度折中。"
    note='''- Prefill Avg为四项百分制准确率等权均值；P+D Avg为两个ROUGE-L%及BLEU的均值，两者不可互比。各均值从未舍入分项计算。
- 两场景独立配置；Balanced/Max-speed是已测代表工作点，不是共同ε保证或全部预算上的全局极值。P+D点为探索性选点；全部旧点仍在原汇总表。
- HAWQ采用当前FP4/INT8/BF16后端的Hessian/ILP PTQ适配；Balanced为r=.80，Max-speed为实际求解的最小可行延迟端点。端点solver索引已与被测method_map逐项验证。
- 相同有效配置共享相同质量/速度；速度按各自campaign的BF16归一化，不是16行全在同轮重测。E2E不含下一批prefill前的恢复，完整cycle另报。
- 粗体为展示精度下该列最高点估计；不据此作统计显著或所有指标全面最优声明。
'''
    body=f'# {model}：rebuttal主表\n\nBaseline与DASO各报告两个场景；HAWQ和DASO各两个代表点。速度相对BF16。\n\n'+table(SHORT)+'\n'+conclusion+'\n\n<details>\n<summary>展开完整Table2任务分项</summary>\n\n'+table(FULL)+'\n</details>\n\n[完整CSV](main_table.csv) · [分组HTML](table.html) · [来源与工作点定义](provenance.json)\n\n<details>\n<summary>实验口径与来源</summary>\n\n'+note+f'\n[原始全部已测点](../{stem}_table2_night161/README.md)\n\n</details>\n'
    (directory/'README.md').write_text(body)
    html_rows=[]
    for r in result:
        cells=[]
        for key,_ in SHORT:
            val=html.escape(r[key]) if key=='method' else f(r[key])
            if key!='method' and float(val)==best[key]:val='<b>'+val+'</b>'
            tag='th' if key=='method' else 'td';cells.append(f'<{tag}>{val}</{tag}>')
        html_rows.append('<tr class="'+('daso' if r['method'].startswith('DASO') else '')+'">'+''.join(cells)+'</tr>')
    page=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{model}</title><style>body{{font:15px system-ui;max-width:1100px;margin:30px auto;padding:0 16px;color:#172330}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:10px;border-bottom:1px solid #dce3e8;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}thead th{{background:#edf2f6}}.daso{{background:#eaf5fc}}.scroll{{overflow:auto}}p,li{{line-height:1.6}}</style><h1>{model}</h1><p>分数与速度越高越好；HAWQ为FP4 PTQ适配，两场景分别配置。</p><div class="scroll"><table><thead><tr><th rowspan="2">方法</th><th colspan="2">Prefill-only</th><th colspan="3">Prefill + Decode</th></tr><tr><th>Avg ↑</th><th>TTFT ×</th><th>Avg ↑</th><th>TPOT ×</th><th>E2E ×</th></tr></thead><tbody>'+''.join(html_rows)+'</tbody></table></div><p><a href="README.md">完整任务与口径</a> · <a href="main_table.csv">CSV</a></p><ul>'+''.join('<li>'+html.escape(line[2:])+'</li>' for line in note.splitlines() if line.startswith('- '))+'</ul></html>'
    (directory/'table.html').write_text(page)
    payload={'model':model,'rows':provenance,'hawq_latency_floor_proof':floor_proof,
        'selection_scope':'Post hoc representative quality-speed reporting; all archived points retained, not a new blind test. Scenario policies separate.'}
    (directory/'provenance.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    TABLES[model]=result
    model_deliveries.append({'model':model,'directory':str(directory.relative_to(BASE)),'rows':16,'metric_cells':16*13})

(OUT/'model_tables_audit.json').write_text(json.dumps({'verified':True,'models':model_deliveries,'source_sha256':SOURCES,
    'new_gpu_measurements':0,'scope':'Readable reporting views only; complete source quality records, map equivalence and Hessian floor indices checked.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verified':True,'models':model_deliveries,'source_files':len(SOURCES)},ensure_ascii=False,indent=2))

# The overview contains readable questions and comparisons; raw policy IDs stay in the source records.
sections=[];overview_tables=[]
def add(text):sections.append(text.strip()+'\n')
def result_table(title,headers,data):
    overview_tables.append({'title':title,'headers':headers,'rows':data})
    return md(headers,data)
def details(text):return '<details>\n<summary>测量说明与来源</summary>\n\n'+text.strip()+'\n\n</details>\n'

add('''# Rebuttal实验结果：清晰版

先看三张**全部baseline＋DASO**主表：[Qwen3-8B](responses/reviewer2/qwen3_main_table/README.md) · [Llama-3.1-8B](responses/reviewer2/llama31_main_table/README.md) · [Llama-2-7B](responses/reviewer2/llama2_main_table/README.md)。它们采用同一展示方式：两个场景各有Avg与速度，HAWQ和DASO各两个代表点。

下面每节先给结论，再给能直接比较的表。正文不放运行编号和调试版本；细节与来源可展开。英文回复单独见[逐reviewer草稿](REBUTTAL_RESPONSE_DRAFT.md)。

默认速度相对BF16，越高越快；标明“相对INT8”的表以INT8为1。Prefill Avg是四项准确率均值，P+D Avg是三项生成分数摘要，两个Avg不互相比大小。
''')
add(result_table('reviewer navigation',['Reviewer','关心的问题','直接查看'],[
 ['R1 / AK6a','实际质量预算、内存开销','[质量预算](#quality-budget)、[运行开销](#runtime-cost)'],
 ['R2 / Exi5','强基线、更多负载、真实混合调度','[模型主表](#model-results)、[不同负载](#workloads)、[混合调度](#mixed)'],
 ['R3 / LrT2','新架构、预测准确性与准备成本','[模型主表](#model-results)、[建模与求解](#model-cost)'],
 ['R4 / QboE','简单剪枝、搜索成本与规模','[剪枝](#pruning)、[建模与求解](#model-cost)、[其他材料](#other)'],
 ['R5 / gkP9','模块接口、反约束优化与方法定位','[接口与反约束](#interfaces-inverse)、[其他材料](#other)']]))

add('''<a id="model-results"></a>
## 1. 模型主结果：补齐强基线后还有收益吗？（R2 / R3）

**结论：** Qwen之外，两套原Llama也存在有竞争力的质量–速度折中。下面保留全部已测常规baseline，能直接与DASO比较；完整任务分项放在各模型主表里。Qwen沿用[刚确认的15行主表](responses/reviewer2/qwen3_main_table/README.md)。
''')
for model,stem,dest,*_ in model_specs:
    data=TABLES[model]
    add('### '+model)
    best={key:max(float(f(r[key])) for r in data) for key,_ in SHORT if key!='method'}
    display=[]
    for r in data:
        line=[]
        for key,_ in SHORT:
            value=r[key] if key=='method' else f(r[key])
            if (key=='method' and r[key].startswith('DASO')) or (key!='method' and float(value)==best[key]):value='**'+value+'**'
            line.append(value)
        display.append(line)
    add(result_table(model,[label for _,label in SHORT],display))
    add(f'[完整任务分项与HTML](responses/reviewer2/{dest}/README.md)')
add(details('''两个场景各自选择配置；Balanced/Max-speed是本表报告的代表点，不是跨所有预算的全局极值或共同实际质量预算。P+D代表点按已测质量–速度折中整理，属于描述性展示。

HAWQ为FP4/INT8/BF16后端的Hessian/ILP PTQ适配；两个工作点分别对应中等延迟目标和最小可行延迟端点。端点求解索引已与实际被测配置核对，相同Prefill配置共享同一组数值。

速度按各自campaign的BF16归一化，不是全表同轮重测。全部旧点保留在[Llama-3.1原汇总](responses/reviewer2/llama31_8b_instruct_table2_night161/README.md)和[Llama-2原汇总](responses/reviewer2/llama2_7b_chat_table2_night161/README.md)。'''))

add('''<a id="quality-budget"></a>
## 2. 质量预算：要求的损失是否真的满足？（R1）

**结论：** 三模型、四个预算和两种验证样本数，共24个预算设定，在确认样本上实测的**平均NLL损失**均未超限。它们复用了14个模型/配置选择，不是24次独立试验。

这里的损失用ΔNLL表示，越小越好；小于允许上限即满足本表的平均损失要求。以下展示64个验证样本时的代表结果。**这些是按NLL预算单独选择的配置，不是上方Balanced/Max-speed的精度声明。**
''')
budgets=records('response174/budget_speed.csv')
add(result_table('measured quality budget',['模型','允许损失上限','实测平均损失','平均损失满足？'],[
 [r['model'],f(r['epsilon']),f(r['confirmation_delta_nll'],4),'是'] for r in budgets]))
add(details('''先用一组样本验证并冻结选择，再用另一组不重叠的64个样本检查；检查结果不参与重新选点。使用公开WikiText中的新窗口，排除旧校准/验证/测试分词流中共享的连续16-token片段，不是外部新语料。

选择规则是验证均值＋2SE＋0.005；上表六个确认均值及均值＋2SE均低于预算。完整24设定中，平均损失全部满足；另加余量的确认保护值并非全部满足（21/24），所以不称为硬质量保证。

这些预算选点可能与模型主表的Balanced/Max-speed不同。NLL是新的确认结果，速度继承同一配置已有正式测量，不是同步新测；NLL预算也不保证每项下游生成分数。

[六个预算与同配置速度](response174/budget_speed.csv) · [完整计数](response174/budget_counts.csv) · [Qwen来源](responses/reviewer1/fresh_quality_run044/README.md) · [Llama-3.1来源](responses/reviewer1/fresh_quality_run045/README.md) · [Llama-2来源](responses/reviewer1/fresh_quality_run046/README.md)。'''))

add('''<a id="workloads"></a>
## 3. 不同负载：优势是否只依赖原配置？（R2 / R3）

**结论：** 固定Qwen主表的Balanced/Max-speed两个P+D策略，六种负载的E2E和完整批次耗时中位数均快于INT8，没有为每行重新调参。

本表速度**相对INT8**。cycle指完成当前批次并恢复到可运行下一批Prefill的总耗时。
''')
work=records('runs/040/results/workloads.csv');wb={(r['workload'],r['policy']):r for r in work}
workdata=[]
for workload in ['center','single','short','long','decode256','single256']:
    a=wb[workload,'w1_e0.01'];b=wb[workload,'w32_e0.03']
    workdata.append([a['batch'],a['prompt'],a['output'],f(a['e2e_ms_vs_int8']),f(b['e2e_ms_vs_int8']),f(a['cycle_ms_vs_int8']),f(b['cycle_ms_vs_int8'])])
add(result_table('six workloads',['批量','输入token','输出token','Balanced E2E×','Max-speed E2E×','Balanced cycle×','Max-speed cycle×'],workdata))
add(details('''所有六个预定负载均展示。点估计为三轮中位数；完整最小/最大范围见图，部分cycle范围跨过1，不解释成每次重复都领先。此结论相对INT8，不声称在全部负载上胜过所有阶段组合；新负载的实际质量另见完整数据，不沿用中心配置的代理参数保证预算。

[全部参考方法和质量](responses/reviewer2/qwen_updated_workloads_run040/README.md) · [可导出PDF图](response174/qwen_six_workloads.pdf) · [SVG](response174/qwen_six_workloads.svg) · [全精度绘图数据](response174/workload_figure_data.csv)。'''))

add('''<a id="mixed"></a>
## 4. 混合调度：同一forward有Prefill和Decode还能用吗？（R2）

**结论：** 新增实现已验证同一次forward中的逐行阶段路由。Qwen Balanced在标准负载达到1.35× BF16，Max-speed在异构长Decode负载达到1.21×。

这是单独的混合调度实验，下面在该实验的九种配置内比较；速度与模型主表分开读取。
''')
mixed=records('responses/reviewer2/qwen_updated_mixed_run047/mixed_results.csv')
mb={(r['case'],r['policy']):r for r in mixed}
method_list=[('BF16','uniform_dense_bf16'),('INT8','uniform_int8'),('SmoothQuant INT8','uniform_sq_int8'),('FP8','uniform_fp8'),
    ('SQ-INT8 → W4A16','fixed_sq_int8_w4a16'),('HAWQ-V3 (Balanced)','hawq_signed_prefill_decode_phase_pair_r0.8'),
    ('HAWQ-V3 (Max-speed)','hardware_minimum_prefill_decode_phase_pair'),('DASO (Balanced)','w1_e0.01'),('DASO (Max-speed)','w32_e0.03')]
standard='chunk1024_b8_s2048_o64';long='chunk1024_b8_heterogeneous_o256'
add(result_table('mixed scheduler',['方法','标准负载 E2E×BF16','异构长Decode E2E×BF16'],[
 [label,f(mb[standard,pid]['e2e_vs_bf16']),f(mb[long,pid]['e2e_vs_bf16'])] for label,pid in method_list]))
add(details('''vLLM V1，chunked prefill预算1024；同一forward内按token行gather/apply/scatter，Prefill和Decode分别使用对应权重。真实mixed计数、各阶段token总数、BF16 no-op一致性及权重一致性均已核验。

使用两份阶段权重常驻的eager变体，代价见下一节内存表。两种场景均为8请求closed-loop burst：标准输入2048/输出64，异构prompt/输出256。它不是生产HTTP/open-loop SLA证明，也不等于主表单arena的Graph路径。

[全部九种配置、实际路由和质量](responses/reviewer2/qwen_updated_mixed_run047/README.md)。'''))

add('''<a id="runtime-cost"></a>
## 5. 运行开销：内存和切换代价多大？（R1 / R5）

### 5.1 两份阶段表示常驻时的内存

**结论：** Qwen Balanced的GPU峰值比BF16低约20%，同时明确付出约8.24 GiB的Host阶段权重缓存。INT8更省GPU内存，DASO提供的是另一种质量–速度–内存折中。
''')
add(result_table('mixed memory',['方法','GPU Torch峰值 GiB','Host阶段权重缓存 GiB'],[
 [label,f(mb[standard,pid]['torch_peak_allocated_gib']),f(mb[standard,pid]['host_adapter_fields_gib'])]
 for label,pid in [('BF16','uniform_dense_bf16'),('INT8','uniform_int8'),('DASO (Balanced)','w1_e0.01'),('DASO (Max-speed)','w32_e0.03')]]))
add(details('''与上一节相同的Qwen混合调度变体、KV6 GiB。GPU峰值是Torch分配峰值，包含KV/workspace等；Host列只是阶段adapter权重缓存，不是总进程RSS。Host/GPU两项不是可以相加的同一内存口径。

[原始内存字段、峰值与协议](responses/reviewer2/qwen_updated_mixed_run047/README.md) · [原主表三模型内存](responses/reviewer1/runtime_memory_night151/README.md)。'''))
add('''### 5.2 把下一批恢复也计入后的速度

**结论：** Qwen两个DASO点计入恢复后，完整cycle仍约为INT8的1.11×。单请求E2E与连续批次的代价分列，避免把恢复成本漏掉。
''')
repeat=records('responses/reviewer2/qwen_full_repeat_run048/repeated_pd.csv');rb={r['policy']:r for r in repeat}
add(result_table('restoration',['方法','E2E ×INT8','完整cycle ×INT8','恢复耗时 ms'],[
 [label,f(rb[pid]['e2e_vs_int8']),f(rb[pid]['cycle_vs_int8']),f(rb[pid]['restore_ms'],1)]
 for label,pid in method_list if pid in ['uniform_dense_bf16','uniform_int8','fixed_sq_int8_w4a16','hardware_minimum_prefill_decode_phase_pair','w1_e0.01','w32_e0.03']]))
add(details('''本表使用主表共同Graph/arena路径的同轮P+D测量，不使用上一节双驻留变体的数据。B8/输入2048/输出64；E2E已包含Prefill→Decode，cycle再计下一批之前的恢复。不能用E2E优势直接代替cycle优势。

[完整重复与恢复数据](responses/reviewer2/qwen_full_repeat_run048/README.md)。'''))

add('''<a id="model-cost"></a>
## 6. 建模与搜索：预测准不准，选策略花多久？（R3 / R4）

### 6.1 未见过的矩阵shape：原Hybrid预测

**结论：** 原Hybrid在shape留出和整组投影维度留出上的MAPE为4.18%–11.83%。MAPE越低，预测越接近实测。
''')
broad=records('responses/reviewer3/overnight_run004/results/broad_latency_validation.csv')
backend_names={'dense_bf16':'Dense BF16','dense_nvfp4':'Dense NVFP4','sparse_bf16':'Sparse BF16','sparse_nvfp4':'Sparse NVFP4','marlin_nvfp4':'W4A16 (FP4)'}
add(result_table('hybrid prediction',['后端','未见shape MAPE ↓','未见投影维度组 MAPE ↓'],[
 [label,f(next(r for r in broad if r['method']==method and r['split']=='shape')['mape_percent'])+'%',
 f(next(r for r in broad if r['method']==method and r['split']=='nk')['mape_percent'])+'%'] for method,label in backend_names.items()]))
add(details('''用历史完整profiling数据重新划分，并仅在各自训练分区拟合；N/K组留出比单独shape留出更严格。这是原设备/协议上的泛化重放，不是新的跨硬件实验。

[全部留出样本、误差与拟合成本](runs/004/REPORT.md)。'''))
add('''### 6.2 已有硬件画像：整模型部署成本估计

**结论：** 对已有画像的策略，加上事先固定的BF16非Linear残差，三个模型两场景的E2E MAPE约1.00%–1.71%。它与上一张未知shape预测表是不同验证。
''')
lut=[]
for label,name in [('Qwen3-8B','qwen016_modeling_points.csv'),('Llama-3.1-8B','llama017_modeling_points.csv'),('Llama-2-7B','llama018_modeling_points.csv')]:
    points=records('responses/reviewer3/modeling_night151/'+name)
    values=[]
    for scene in ['prefill_only','prefill_decode']:
        group=[r for r in points if r['scenario']==scene and r.get('measured_e2e_ms') and r.get('absolute_percentage_error')]
        values.append(f(sum(float(r['absolute_percentage_error']) for r in group)/len(group))+'%')
    lut.append([label,*values])
add(result_table('deployment prediction',['模型','Prefill MAPE ↓','P+D MAPE ↓'],lut))
add(details('''各Linear画像已经包含内部激活量化、scale/metadata处理和GEMM；attention、KV、norm/activation及引擎等非Linear工作，用分配前固定的BF16残差近似覆盖，不假设为零。更换负载不能直接外推相同残差，更换硬件需要重新画像/校准。

Qwen覆盖43个已测策略场景；Llama-3.1覆盖31个核心场景，Llama-2覆盖29个，未测点不混入误差。阶段恢复另报cycle。

[三模型逐点预测与PDF/SVG](responses/reviewer3/modeling_night151/README.md)。'''))
add('''### 6.3 已校准后，重新选一次策略

**结论：** 两步精确代理求解合计约0.2–2.0秒，修改预算或偏好时可复用已有画像与质量模型。这里测的是**求解本身**。
''')
solve=records('responses/reviewer3/new_point_overheads_night161/snapshots/20261027_184651_final/solver_points.csv')
solve_data=[]
for label,runs in [('Qwen3-8B',['032','033']),('Llama-3.1-8B',['034','036']),('Llama-2-7B',['035'])]:
    values=[float(r['total_solve_seconds']) for r in solve if r['run'] in runs]
    solve_data.append([label,f(min(values),3)+'–'+f(max(values),3)+' s'])
add(result_table('solver cost',['模型','一次两步求解耗时范围'],solve_data))
add('''### 6.4 新增8bit后的准备增量

复用原完整策略标签时，两套Llama的增量准备如下；从零标签采集、导出与验证仍需另计，不把上面的几秒求解当作全部部署成本。
''')
prep=records('responses/reviewer3/new_point_overheads_night161/snapshots/20261027_184651_final/preparation_reference.csv')
pb={(r['run'],r['stage']):float(r['internal_seconds']) for r in prep}
add(result_table('incremental preparation',['阶段','Llama-3.1耗时 s','Llama-2耗时 s'],[
 ['新增8bit局部特征',f(pb['017','new_8bit_features']),f(pb['018','new_8bit_features'])],
 ['80行Linear硬件画像',f(pb['017','linear_profile_80_rows']),f(pb['018','linear_profile_80_rows'])],
 ['两场景质量模型拟合',f(pb['017','quality_fit_prefill']+pb['017','quality_fit_pd']),f(pb['018','quality_fit_prefill']+pb['018','quality_fit_pd'])]]))
add(details('''求解时间为两次HiGHS内部wall time：先最小化预测延迟，再在同速条件下最小化预测质量损失；优化的是固定加和代理。准备表为函数内部计时。进程壁钟之和、并行总墙钟和GPU-active时间是不同口径，不混加。

[完整新点开销](responses/reviewer3/new_point_overheads_night161/snapshots/20261027_184651_final/README.md) · [标签/Hessian/导出/验证等完整成本](night151/COST_REPORT.md)。'''))

add('''<a id="pruning"></a>
## 7. 剪枝候选：SparseGPT比简单剪枝更合适吗？（R4）

**结论：** 同2:4结构、同Sparse BF16 kernel下，SparseGPT在MMLU和WikiText PPL上优于Wanda与Magnitude，支持它作为DASO的稀疏候选。相近速度主要来自共享kernel。
''')
prune=records('responses/reviewer4/pruning_llama31_run020/pruning_table.csv')
add(result_table('pruning comparison',['方法','MMLU ↑','WikiText PPL ↓','Prefill ×BF16','P+D E2E ×BF16'],[
 [r['method'],f(r['mmlu']),f(r['wikitext_word_ppl']),f(r['prefill_ttft_speedup']),f(r['pd_e2e_speedup'])] for r in prune]))
add(details('''Llama-3.1-8B-Instruct的完整任务评测。PPL来自完整lm-eval word perplexity，不是少量NLL窗口取exp。Wanda采用层序传播，按原记录重建128×2048训练样本；原SparseGPT tensor未保留，因此不声称字节完全一致。

这里只对表列指标下结论；完整生成任务中Wanda在部分指标更高，全部原始结果保留。[完整质量、速度和剪枝准备成本](responses/reviewer4/pruning_llama31_run020/README.md)。'''))

add('''<a id="matched-allocation"></a>
## 8. 分配方法：同候选、同约束下质量代理有用吗？（R1 / R2 / R3）

**结论：** 在Llama-3.1的一个相同延迟上限对照中，DASO在近似相同实测延迟下得到更低的平均NLL损失。候选集合和求解器相同，区别是质量代理。
''')
matched=records('responses/reviewer2/llama31_matched_allocator_run041/matched_points.csv')
control=[r for r in matched if r['scenario']=='prefill_decode' and r['phase_mode']=='phase_pair' and float(r['r'])==.8]
add(result_table('matched allocator',['方法','实测平均NLL损失 ↓','实测E2E ms ↓','共同E2E上限 ms'],[
 ['DASO' if r['allocator']=='daso' else 'Hessian / ILP',f(r['delta_nll'],5),f(r['e2e_ms']),f(r['original_target_ms'])] for r in control]))
add(details('''双方使用BF16/INT8/NVFP4相同候选和MILP，P+D阶段独立。比较对象为signed-Hessian适配；不把该单点结论推广为胜过所有Hessian变体。

NLL复用原冻结64个窗口，速度同轮补齐。DASO−Hessian的paired差值为−0.00449±0.00361（2SE），属于已研究配置的描述性结果。

[全部同候选对照与配对误差](responses/reviewer2/llama31_matched_allocator_run041/README.md)。'''))

add('''<a id="interfaces-inverse"></a>
## 9. 反约束与接口：能满足延迟上限、连接不同格式吗？（R5）

### 9.1 反约束与实测延迟检查

**结论：** 把问题改成“延迟不超过上限时，让质量损失最小”可以执行；已有固定代理一致性和真实运行测量支持。
''')
inverse=read('runs/023/results/inverse_checks.json');assert len(inverse)==8 and all(r['passed'] for r in inverse)
gate=read('night161/latency_validation/final_audit.json');assert gate['verified']
gate_rows=records('night161/latency_validation/latency_validation.csv')
assert all(int(r['selected_misses'])==0 for r in gate_rows)
assert len(matched)==18 and all(int(r['e2e_target_misses'])==0 for r in matched)
add(result_table('inverse and latency',['检查内容','检查范围','实际结果'],[
 ['正向/反向求解一致性','8个固定代理问题','8/8通过'],
 ['原绝对延迟上限','18套配置，每套24次计时','0次观察超时'],
 ['首轮校验后检查后两轮','36个目标设定，复用17套配置','36个均可行，后两轮0次观察超时']]))
add(details('''正反求解检查只针对固定加和代理，不能替代真实质量评估。18套配置包含DASO及Hessian，覆盖三个延迟比例和Prefill/P+D。

36个目标设定来自另一个预先冻结规则的分割分析：round0最大样本×1.05作筛选，round1/2每目标共16个timed观察。E2E和cycle分别定义cap；原对应点在这些cap下也无超时，所以不宣称校验降低了本批违约。分析在三轮收集完成后执行，但规则先冻结，选择先于后两轮数据读取封存；不是在线干预，也不是生产硬SLA保证。

[正反代理检查](responses/reviewer5/solver_grid_run023/README.md) · [原绝对上限实测](responses/reviewer2/llama31_matched_allocator_run041/README.md) · [预注册分割校验](responses/reviewer5/latency_validation_night161/README.md)。'''))
add('''### 9.2 不同压缩后端之间的模块接口

**结论：** 已测768条Linear边界均为BF16输入/输出，内部压缩权重格式不会直接变成下一模块必须接收的activation格式。
''')
interfaces=records('responses/reviewer5/interfaces_run025/interfaces.csv')
interface_rows=[]
for pid,label in [('hardware_minimum_prefill_decode_phase_pair','INT8/FP4阶段混合配置'),('uniform_sq_int8','SmoothQuant INT8'),('uniform_fp8','FP8')]:
    group=[r for r in interfaces if r['policy']==pid]
    assert len(group)==256 and all(json.loads(r['input_dtype'])=='torch.bfloat16' and json.loads(r['output_dtype'])=='torch.bfloat16' for r in group)
    interface_rows.append([label,len(group),'BF16','BF16'])
add(result_table('module interfaces',['配置','实测边界条数','输入','输出'],interface_rows))
add(details('''Llama-3.1三套配置，各128 Linear×2阶段；短输入真实forward记录dtype/shape/stride。内部激活量化、scale和metadata已计入Linear画像；阶段权重搬运与恢复另报，不能由接口dtype一致推断总转换开销为零。

[768条实际记录与接口审计](responses/reviewer5/interfaces_run025/README.md) · [连续MLP方法对检查](runs/004/results/mlp_interfaces.csv)。'''))

add('''<a id="other"></a>
## 10. 其他已备妥材料与仍需限定的范围

- **Figure2乱码：** 已有无字体依赖的[替换PDF](runs/008/results/figure2_outlined.pdf)和[SVG](runs/008/results/figure2_outlined.svg)，保留原公式含义。
- **相同输入如何测：** 从BF16参考模型捕获同一个模块输入，分别调用各候选；候选输出不替换参考主干继续传播的activation。[具体说明](responses/reviewer5/overnight_run008/IDENTICAL_INPUTS.md)。
- **NAS方法定位：** 聚焦固定LLM的跨后端、模块、阶段联合选择和实际部署；承认资源约束搜索与硬件预测已有先例。[定位草稿](runs/008/NAS_CONTEXT.md)。
- **更大模型规模：** 已有112个Qwen3-32B维度的真实kernel forward，支持更大矩阵的后端可执行性；尚无完整30/32B模型质量/E2E结果。[规模测量](runs/004/REPORT.md)。

完整30/32B部署及生产open-loop在线SLA仍需另补证据；当前的kernel测量和mixed burst均按各自范围使用。

[精简英文回复](REBUTTAL_RESPONSE_DRAFT.md) · [本次来源与检查](results176/audit.json) · [重排前的完整旧版](results176/reference/REBUTTAL_SELECTED_EVIDENCE.md)
''')

text='\n'.join(sections)
(BASE/'REBUTTAL_SELECTED_EVIDENCE.md').write_text(text)
(OUT/'overview_tables.json').write_text(json.dumps(overview_tables,ensure_ascii=False,indent=2)+'\n')
(OUT/'overview_sources.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':SOURCES,
    'new_gpu_measurements':0,'models':model_deliveries,'overview_table_count':len(overview_tables),
    'scope':'Reorganized reporting; no original experiments changed. Display names are mapped to actual per-experiment policies.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'overview_lines':len(text.splitlines()),'overview_tables':len(overview_tables),'source_files':len(SOURCES)},ensure_ascii=False,indent=2))
