"""Publish one Qwen comparison table, with explicit per-scenario policy bindings."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
REPO=BASE.parent.parent
SOURCES={}

def track(path):
    path=Path(path)
    SOURCES[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read(relative):
    return json.loads(track(BASE/relative).read_text())

def csv_rows(relative):
    return list(csv.DictReader(track(BASE/relative).open()))

def signature(mapping):
    return hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()

def policy(run,pid):
    return read(f'runs/{run}/configs/policies/{pid}.json')['method_map']

master_path='responses/reviewer2/qwen3_table2_night161/main_table.csv'
master={r['method']:r for r in csv_rows(master_path)}
old_prov={r['method']:r for r in read('responses/reviewer2/qwen3_table2/provenance.json')['rows']}
ext_prov={r['method']:r for r in read('responses/reviewer2/qwen3_table2_run016/provenance.json')['new_rows']}
repeat_path='responses/reviewer2/qwen_full_repeat_run048/repeated_pd.csv'
repeat={r['policy']:r for r in csv_rows(repeat_path)}
pp_fields=['arc_c','arc_e','wino','mmlu','prefill_ttft_speedup']
pd_fields=['cnn_dm','dialogsum','iwslt','pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']


def validate_quality(cells,values,scenario):
    result={}
    for field in (pp_fields[:4] if scenario=='prefill_only' else pd_fields[:3]):
        cell=cells[field]
        path=track(REPO/cell['source'])
        raw=json.loads(path.read_text())
        if scenario=='prefill_only':
            assert raw['limit'] is None and raw['num_fewshot']==0
            actual=float(raw['metrics'][cell['metric']])*100
            counts=raw['raw_lm_eval']['n-samples']
            assert sum(r['effective'] for r in counts.values())==cell['samples']
            assert all(r['original']==r['effective'] for r in counts.values())
        else:
            actual=float(raw[cell['metric']])
            assert raw['samples']==cell['samples'] and raw['max_new_tokens']==256
        assert math.isclose(actual,values[field],rel_tol=0,abs_tol=1e-10),(field,actual,values[field])
        result[field]={**cell,'sha256':SOURCES[str(path.relative_to(REPO))]}
    return result


def legacy(method,scenario):
    fields=pp_fields if scenario=='prefill_only' else pd_fields
    values={key:float(master[method][key]) for key in fields}
    record=old_prov[method]
    quality=validate_quality(record['cells'],values,scenario)
    entry=record['policies'][scenario]
    mapping=policy('015',entry['policy'])
    source_mapping=policy(*entry['quality'])
    if scenario=='prefill_only':
        assert {n:m['prefill_method'] for n,m in mapping.items()}=={n:m['prefill_method'] for n,m in source_mapping.items()}
    else:
        assert mapping==source_mapping
    return values,{'scenario':scenario,'policy_run':'015','policy':entry['policy'],'map_sha256':signature(mapping),
        'source_table':master_path,'source_row':method,'quality':quality,
        'speed':{key:record['cells'][key] for key in fields if key.endswith('_speedup')}}


def prefill16(epsilon):
    method=f'DASO-8 incl. plain INT8 + exact/validation, epsilon={epsilon}'
    values={key:float(master[method][key]) for key in pp_fields}
    record=ext_prov[method]
    quality=validate_quality(record['prefill_only_quality_sources'],values,'prefill_only')
    pid=record['policies']['prefill_only']
    mapping=policy('016',pid)
    selected=next(r for r in read('runs/016/configs/selected_policies.json')['selected']
        if r['scenario']=='prefill_only' and math.isclose(r['quality_budget'],float(epsilon)))
    assert selected['policy']==pid
    return values,{'scenario':'prefill_only','policy_run':'016','policy':pid,'map_sha256':signature(mapping),
        'source_table':master_path,'source_row':method,'quality':quality,
        'speed':{'prefill_ttft_speedup':values['prefill_ttft_speedup'],'range':record['speed_ranges']['prefill_ttft_speedup']},
        'selection':selected}


def repeated(pid):
    raw=repeat[pid]
    values={key:float(raw[key]) for key in pd_fields}
    cells={}
    for field,dataset,metric,count in [('cnn_dm','cnn_dm_1000','rougeL_percent',1000),
        ('dialogsum','dsum','rougeL_percent',1500),('iwslt','IWSLT','sacre_bleu',333)]:
        cells[field]={'source':f'artifacts/rebuttal/runs/048/results/generation/{pid}/{dataset}/metrics_0000_end.json',
            'metric':metric,'samples':count}
    quality=validate_quality(cells,values,'prefill_decode')
    return values,{'scenario':'prefill_decode','policy_run':'048','policy':pid,
        'map_sha256':signature(policy('048',pid)),'source_table':repeat_path,'quality':quality,
        'speed':{key:values[key] for key in pd_fields if key.endswith('_speedup')},
        'e2e_vs_int8_range':[float(raw['e2e_vs_int8_min']),float(raw['e2e_vs_int8_max'])]}

# HAWQ speed endpoint must actually be an archived Hessian/ILP solution at the latency floor.
pool=read('runs/015/configs/reanalysis_pool.json')
endpoint={}
for scene in ['prefill_only','prefill_decode']:
    point=next(p for p in pool['points'] if p['label']==f'graph_hawq_fp4_{scene}_q0.07')
    assert point['family']=='fresh HAWQ PTQ adaptation'
    assert math.isclose(point['predicted_linear_ms'],point['minimum_linear_ms'],abs_tol=1e-6)
    assert abs(point['linear_limit_ms']-point['minimum_linear_ms'])<2e-7
    endpoint[scene]=point
assert policy('015',endpoint['prefill_only']['policy_id'])==policy('016','hardware_minimum_prefill_only_same_operator')
assert policy('015',endpoint['prefill_decode']['policy_id'])==policy('048','hardware_minimum_prefill_decode_phase_pair')

rows=[];provenance=[]
def add(method,pp,pd,point_definition=None):
    a,pa=pp;b,pb=pd
    row={'model':'Qwen3-8B','method':method,**a,**b}
    row['prefill_avg_acc']=sum(row[k] for k in pp_fields[:4])/4
    row['pd_avg_score']=sum(row[k] for k in pd_fields[:3])/3
    rows.append(row)
    provenance.append({'method':method,'prefill_only':pa,'prefill_decode':pb,'point_definition':point_definition})

baselines=[('BF16','BF16','uniform_dense_bf16'),('Dense NVFP4','Dense NVFP4',None),
    ('W4A16 (FP4)','W4A16 NVFP4',None),('W4A16 (INT4)','W4A16 INT4 Marlin',None),
    ('Sparse BF16 (2:4)','Sparse BF16',None),('Sparse NVFP4 (2:4)','Sparse NVFP4',None),
    ('INT8 (W8A8)','INT8 W8A8','uniform_int8'),('SmoothQuant INT8','SmoothQuant INT8','uniform_sq_int8'),
    ('FP8 (W8A8)','FP8 W8A8','uniform_fp8'),('SmoothQuant FP8','SmoothQuant FP8','uniform_sq_fp8')]
for display,old,pid in baselines:
    pp=prefill16('0.01') if pid=='uniform_int8' else legacy(old,'prefill_only')
    pd=repeated(pid) if pid else legacy(old,'prefill_decode')
    add(display,pp,pd)
phase_pp=legacy('SmoothQuant INT8','prefill_only')
phase_pd=repeated('fixed_sq_int8_w4a16')
assert {n:m['prefill_method'] for n,m in policy('015',phase_pp[1]['policy']).items()}=={
    n:m['prefill_method'] for n,m in policy('048','fixed_sq_int8_w4a16').items()}
add('SQ-INT8 → W4A16',phase_pp,phase_pd,'Fixed uniform prefill/decode baseline.')
add('HAWQ-V3 (Balanced)',legacy('HAWQ-v3 FP4, latency target=.80 BF16 (PTQ)','prefill_only'),
    repeated('hawq_signed_prefill_decode_phase_pair_r0.8'),
    'FP4/INT8/BF16 PTQ adaptation; Hessian objective under r=.80 latency cap, separately for both scenarios.')
shared=prefill16('0.04')
add('HAWQ-V3 (Max-speed)',shared,repeated('hardware_minimum_prefill_decode_phase_pair'),
    {'definition':'Archived Hessian/ILP output at the minimum feasible Linear-latency cap; not the infeasible original request.',
     'endpoint_solver_records':endpoint,'shared_prefill_measurement':'Identical full method_map to DASO speed point; reuse one measurement.'})
pp_balanced=legacy('SmoothQuant INT8','prefill_only')
selected=next(p for p in read('runs/015/configs/selected_reanalysis.json')['selected']
    if p['scenario']=='prefill_only' and p['quality_budget']==.01)
assert selected['policy_id']=='uniform_sq_int8'
pp_balanced[1]['selection']=selected
add('DASO (Balanced)',pp_balanced,repeated('w1_e0.01'),
    'Representative quality-oriented reported point; prefill selected SQ-INT8, P+D frozen D1; no common actual epsilon claimed.')
add('DASO (Max-speed)',shared,repeated('w32_e0.03'),
    'Representative speed-oriented reported point; prefill minimum-cost incumbent, P+D frozen D32. Faster displayed point, not a global maximum over all archived budgets.')
assert len(rows)==15
columns=['model','method','arc_c','arc_e','wino','mmlu','prefill_avg_acc','prefill_ttft_speedup',
    'cnn_dm','dialogsum','iwslt','pd_avg_score','pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']
assert all(math.isfinite(r[key]) for r in rows for key in columns[2:])
with (HERE/'main_table.csv').open('w') as handle:
    writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
compact=[('method','方法'),('prefill_avg_acc','Prefill Avg ↑'),('prefill_ttft_speedup','Prefill TTFT ×'),
    ('pd_avg_score','P+D Avg ↑'),('pd_ttft_speedup','P+D TTFT ×'),('pd_tpot_speedup','P+D TPOT ×'),('pd_e2e_speedup','P+D E2E ×')]
full=[('method','方法'),('arc_c','ARC-C'),('arc_e','ARC-E'),('wino','Wino'),('mmlu','MMLU'),
    ('prefill_avg_acc','Prefill Avg'),('prefill_ttft_speedup','Prefill TTFT ×'),('cnn_dm','CNN/DM'),
    ('dialogsum','DialogSum'),('iwslt','IWSLT'),('pd_avg_score','P+D Avg'),('pd_ttft_speedup','P+D TTFT ×'),
    ('pd_tpot_speedup','P+D TPOT ×'),('pd_e2e_speedup','P+D E2E ×')]
best={key:max(float(f'{r[key]:.2f}') for r in rows) for key in columns[2:]}
def md_table(fields):
    lines=['| '+' | '.join(label for _,label in fields)+' |','| '+' | '.join('---' if key=='method' else '---:' for key,_ in fields)+' |']
    for row in rows:
        cells=[]
        for key,_ in fields:
            value=row[key] if key=='method' else f'{row[key]:.2f}'
            if (key=='method' and row['method'].startswith('DASO')) or (key!='method' and float(value)==best[key]):
                value='**'+value+'**'
            cells.append(value)
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)
by={r['method']:r for r in rows}
gain=100*(by['DASO (Max-speed)']['pd_e2e_speedup']/by['HAWQ-V3 (Max-speed)']['pd_e2e_speedup']-1)
quality_gain=by['DASO (Max-speed)']['pd_avg_score']-by['HAWQ-V3 (Max-speed)']['pd_avg_score']
notes='''- Prefill Avg是ARC-C、ARC-E、Wino、MMLU四项百分制准确率的等权均值；P+D Avg是CNN/DM与DialogSum的ROUGE-L%及IWSLT BLEU的等权摘要，两种Avg不可互比。所有均值由未舍入分项计算。
- B=8、输入2048 token；Prefill-only测O=1的TTFT，P+D测O=64。每行两场景各有独立配置，与论文Table2口径一致；P+D TTFT不作为独立Prefill-only结果。
- Balanced/Max-speed是本表报告的两个代表工作点，按场景独立选择；不代表跨全部预算的全局最优。HAWQ为适配当前FP4/INT8/BF16后端的Hessian/ILP PTQ实现，Balanced用r=.80延迟目标，Max-speed用最小可行Linear延迟端点。
- HAWQ与DASO的Prefill Max-speed是同一有效配置，因此共用同一组精度和最新对应速度。DASO Prefill Balanced选中SQ-INT8，也与该uniform baseline共用数值。
- 这是一张已完成实验的汇总表，各速度按各自campaign的BF16归一化，并非15行全部同轮重测。主要P+D强基线和两个DASO点采用048同轮重复；其余原uniform点保留015结果。完整来源与point-ID放在provenance.json，主表不堆叠历史版本。
- 粗体表示按展示精度该列最高的点估计，并列值同样加粗；DASO方法名另加粗便于定位。E2E包括prefill→decode，下一批恢复另计cycle；本表不将速度差或Avg点估计差称为统计显著结论。
'''
intro='''# Qwen3-8B：rebuttal主表

主表仅保留baseline与DASO；HAWQ-V3和DASO各两个工作点。分数及加速比越高越好，速度相对BF16。HAWQ-V3为FP4 PTQ硬件适配；两场景分别使用各自配置。
'''
body=intro+'\n'+md_table(compact)+f'''\n\n**可以直接读出的优势：** Prefill Balanced下，DASO为70.58 / 1.89×，HAWQ为70.47 / 1.32×；Prefill Max-speed两者持平。P+D Max-speed下，DASO的Avg为29.95、E2E为2.03×，HAWQ为29.75、1.91×，即E2E相对更快约{gain:.1f}%，Avg点估计高{quality_gain:.2f}。

<details>
<summary>展开完整Table2：四项Prefill任务、三项生成任务和两场景速度</summary>

'''+md_table(full)+'''

</details>

[分组列HTML表](table.html) · [完整精度CSV](main_table.csv) · [逐场景配置与来源](provenance.json)

<details>
<summary>口径与两个工作点的定义</summary>

'''+notes+'''
</details>
'''
(HERE/'README.md').write_text(body)

# HTML makes the two scenario groups explicit without displaying experimental history.
def html_table(fields,grouped=False):
    head=''
    if grouped:
        head='<tr><th rowspan="2">方法</th><th colspan="6">Prefill-only</th><th colspan="7">Prefill + Decode</th></tr>'
        head+='<tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in fields[1:])+'</tr>'
    else:
        head='<tr><th rowspan="2">方法</th><th colspan="2">Prefill-only</th><th colspan="4">Prefill + Decode</th></tr>'
        head+='<tr>'+''.join('<th>'+html.escape(label.replace('Prefill ','').replace('P+D ',''))+'</th>' for _,label in fields[1:])+'</tr>'
    data=[]
    for row in rows:
        cls='daso' if row['method'].startswith('DASO') else 'hawq' if row['method'].startswith('HAWQ') else ''
        cells=[]
        for key,_ in fields:
            value=html.escape(row[key]) if key=='method' else f'{row[key]:.2f}'
            if key!='method' and float(value)==best[key]:value='<strong>'+value+'</strong>'
            cells.append('<'+('th' if key=='method' else 'td')+'>'+value+'</'+('th' if key=='method' else 'td')+'>')
        data.append('<tr class="'+cls+'">'+''.join(cells)+'</tr>')
    return '<div class="scroll"><table><thead>'+head+'</thead><tbody>'+''.join(data)+'</tbody></table></div>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Qwen3 Rebuttal 主表</title><style>
body{font-family:system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 18px;color:#172330;background:#fff}h1{font-size:26px}p{line-height:1.65}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:15px}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #dfe5e9;white-space:nowrap}th:first-child{text-align:left}thead th{background:#edf2f6;text-align:center}tbody th{font-weight:500}.daso{background:#eaf5fc}.daso th{font-weight:700}.hawq{background:#fff9ed}details{margin:24px 0}summary{cursor:pointer;font-weight:600}a{color:#125f91}li{line-height:1.65;margin:8px 0}strong{font-weight:750}
@media print{body{max-width:none;margin:0}details{break-inside:avoid}table{font-size:10px}th,td{padding:5px}}
</style><h1>Qwen3-8B · Rebuttal 主表</h1><p>Baseline与DASO的两个场景、两个报告工作点。分数与速度均越高越好。HAWQ-V3为FP4 PTQ硬件适配；两个场景分别配置。</p>'''
page+=html_table(compact)
page+=f'<p><strong>P+D速度优先点：</strong>DASO 2.03× / Avg 29.95，HAWQ 1.91× / Avg 29.75；E2E相对更快约{gain:.1f}%。</p>'
page+='<details><summary>完整Table2任务分项</summary>'+html_table(full,True)+'</details>'
page+='<p><a href="main_table.csv">下载完整CSV</a> · <a href="README.md">Markdown</a> · <a href="provenance.json">来源与配置</a></p>'
page+='<details><summary>实验口径</summary><ul>'+''.join('<li>'+html.escape(line[2:])+'</li>' for line in notes.splitlines() if line.startswith('- '))+'</ul></details></html>'
(HERE/'table.html').write_text(page)

expected_manifests={'015':'f2373064e52549bdb23fe4748df4130c4df6d5dee90637b6929eed08df5e19b1',
    '016':'7366eec8710a81650875c630c85ba679b8b1116a4d426d1912e44b042286e334'}
for rid,expected in expected_manifests.items():
    read(f'runs/{rid}/results/manifest.json')
    assert SOURCES[f'artifacts/rebuttal/runs/{rid}/results/manifest.json']==expected
final48=read('runs/048/results/final_audit.json')
assert final48['complete'] and final48['verified']
# Source manifests themselves are hashed; their complete file payloads remain unchanged.
provenance_data={'created_utc':datetime.now(timezone.utc).isoformat(),'rows':provenance,'source_sha256':SOURCES,
    'derived_metrics':{'prefill_avg_acc':'arithmetic mean of four unrounded percentage accuracies',
        'pd_avg_score':'arithmetic mean of two ROUGE-L percentages and corpus BLEU; descriptive summary'},
    'scope':'Curated complete Qwen table, independent scenario configurations and cross-campaign BF16 normalization; not all rows remeasured together.',
    'reporting_point_scope':'Balanced/Max-speed label the two displayed representative points; no common actual epsilon or global optimum claim.',
    'hawq_endpoint_verified':True,'shared_prefill_map_verified':True,'prefill_balanced_selection_source':'run015 verified SQ-INT8 point',
    'parent_manifests_unchanged':expected_manifests}
(HERE/'provenance.json').write_text(json.dumps(provenance_data,ensure_ascii=False,indent=2)+'\n')
audit={'verified':True,'rows':len(rows),'original_metric_cells':15*11,'derived_avg_cells':15*2,
    'total_metric_cells':15*13,'quality_source_records_verified':sum(len(r[s]['quality']) for r in provenance for s in ['prefill_only','prefill_decode']),
    'finite_metrics':True,'averages_from_unrounded':True,'hawq_latency_floor_verified':True,'shared_prefill_result_exact':True,
    'new_gpu_measurements':0,'source_files_hashed':len(SOURCES),
    'positive_comparisons':{'daso_vs_hawq_speed_pd_e2e_gain_percent':gain,'pd_avg_point_difference':quality_gain},
    'output_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['README.md','table.html','main_table.csv','provenance.json']}}
(HERE/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False,indent=2))
