"""Combine the preserved 17-row table and two new complete run016 rows."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = REPO / 'artifacts/rebuttal/runs/016'
OUTPUT = REPO / 'artifacts/rebuttal/responses/reviewer2/qwen3_table2_run016'
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def main():
    assert read(RUN/'results/final_audit.json')['verified']
    original=REPO/'artifacts/rebuttal/responses/reviewer2/qwen3_table2/main_table.csv'
    SOURCES[str(original.relative_to(REPO))]=hashlib.sha256(original.read_bytes()).hexdigest()
    rows=list(csv.DictReader(original.open()))
    columns=list(rows[0])
    for row in rows:
        for key in columns[2:]:row[key]=float(row[key])
    assert len(rows)==17
    selection=read(RUN/'configs/selected_policies.json')['selected']
    metrics={(x['label'],x['scenario']):x for x in read(RUN/'results/full_task_metrics.json')}
    evidence=[]
    for epsilon in [.01,.04]:
        row={'model':'Qwen3-8B','method':f'DASO-8 incl. plain INT8 + exact/validation, epsilon={epsilon:g}'}
        trace={'method':row['method'],'run':'016','policies':{},'speed_ranges':{}}
        for scene in ['prefill_only','prefill_decode']:
            choice=next(x for x in selection if x['scenario']==scene and x['quality_budget']==epsilon)
            pid=choice['policy'];trace['policies'][scene]=pid
            full=metrics[choice['label'],scene]
            row.update(full['metrics']);trace[scene+'_quality_sources']=full['sources']
            output=1 if scene=='prefill_only' else 64
            keys=['e2e_ms'] if output==1 else ['ttft_ms','tpot_ms','e2e_ms']
            ratios={key:[] for key in keys}
            for campaign in range(3):
                directory=RUN/f'results/speed/node0/round{campaign}'
                samples=[s for c in read(directory/f'{pid}.json')['cases'] if c['output']==output for s in c['samples'] if not s['warmup']]
                anchors=[s for name in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']
                    for c in read(directory/f'{name}.json')['cases'] if c['output']==output
                    for s in c['samples'] if not s['warmup']]
                assert len(samples)==8 and len(anchors)==16
                for key in keys:ratios[key].append(statistics.median(s[key] for s in anchors)/
                    statistics.median(s[key] for s in (anchors if pid=='uniform_dense_bf16' else samples)))
            for key,values in ratios.items():
                column='prefill_ttft_speedup' if output==1 else 'pd_'+key.replace('_ms','_speedup')
                row[column]=statistics.median(values)
                trace['speed_ranges'][column]={'min':min(values),'max':max(values)}
        assert set(row)==set(columns)
        rows.append(row);evidence.append(trace)
    for row in rows:
        row['prefill_avg_acc']=statistics.mean(row[key] for key in ['arc_c','arc_e','wino','mmlu'])
        row['pd_avg_score']=statistics.mean(row[key] for key in ['cnn_dm','dialogsum','iwslt'])
    columns.insert(columns.index('prefill_ttft_speedup'),'prefill_avg_acc')
    columns.insert(columns.index('pd_ttft_speedup'),'pd_avg_score')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'main_table.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    lines=['# Qwen3-8B完整Table 2：17行原Graph表＋2行DASO-8（R1/R2/R3/R4/R5）','',
        '**19行、247项指标完整（含38项派生均值）。** 前17行原指标逐值保留run015主表；后两行来自已完成并封存的run016。'
        '这是跨两次实验的完整汇总，不称为19行同一campaign重测。每行速度按各自三轮BF16首尾锚点归一化；'
        'run016新行的完整任务全部重新计算。原CSV未改动。','',
        '共同负载RTX5090 GPU0/NUMA0、B8/S2048/O1或64、6GiB BF16 KV、native decode Graph；'
        '三轮3warmup+8timed。ARC-C acc_norm、ARC-E/Wino/MMLU acc；CNN/DM1000、DialogSum1500、IWSLT333。'
        '完整生成最多256新token且允许EOS；TTFT/TPOT/E2E测速固定输出长度另行执行。','',
        'Prefill Avg acc为ARC-C、ARC-E、Wino、MMLU的等权算术平均；P+D Avg score为CNN/DM、DialogSum、IWSLT分数的等权算术平均。'
        '均使用未舍入分项计算，不按数据集样本数加权。生成平均分混合ROUGE-L%与SacreBLEU，仅用于汇总展示，不作为统一准确率或NLL质量预算。','',
        '| Method | ARC-C | ARC-E | Wino | MMLU | Prefill Avg acc | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D Avg score | P+D TTFT × | TPOT × | E2E × |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:lines.append('| '+row['method']+' | '+' | '.join(f'{row[key]:.2f}' for key in columns[2:])+' |')
    lines += ['', '最后两行的Prefill和P+D分别选择各自配置。ε=.01：Prefill选择plain INT8，P+D回退BF16；'
        'ε=.04：Prefill为硬件端点，P+D为独立阶段的八候选配置。候选/训练扩展、精确求解及validation是额外流程，'
        '不能全部归因为原DP或质量代理优势。','',
        '新DASO-8 ε=.04的P+D更快，但三个完整生成指标低于强HAWQ对照；质量预算仅针对固定NLL，'
        '不构成跨下游任务保证。旧七候选版本和不利结果保留。纯分配器比较请用同候选/成本/上限对照。','',
        '[CSV](main_table.csv) · [逐项来源](provenance.json) · [旧17行表](../qwen3_table2/README.md) · '
        '[同候选完整对照](../matched_allocator_run016/README.md) · [run016原始记录](../../../runs/016/README.md)','']
    (OUTPUT/'README.md').write_text('\n'.join(lines))
    (OUTPUT/'provenance.json').write_text(json.dumps({'complete':True,'rows':19,'metric_cells':247,
        'original_metric_cells':209,'derived_metric_cells':38,
        'derived_metrics':{
            'prefill_avg_acc':{'operation':'unweighted arithmetic mean of unrounded scores','inputs':['arc_c','arc_e','wino','mmlu']},
            'pd_avg_score':{'operation':'unweighted arithmetic mean of unrounded scores','inputs':['cnn_dm','dialogsum','iwslt'],
                'scope':'Mixed ROUGE-L percent and SacreBLEU summary; not a common accuracy measure or NLL constraint.'}},
        'old_rows_source':str(original.relative_to(REPO)),'old_rows_run':'015','new_rows':evidence,
        'sources_sha256':SOURCES,'scope':'Cross-campaign complete overview; retain per-campaign normalization and original source provenance.'},indent=2)+'\n')
    print('Published complete Qwen overview: 19 rows, 247 cells including 38 derived averages',flush=True)


if __name__=='__main__':main()
