"""Publish all fixed-map transfers, with actual per-load quality and cycle."""
import csv
import hashlib
import json
import math
import shutil
import statistics

from common import RUN, REPO, write_json
from launch import read


def save(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    assert read(RUN/'results/quality_complete.json')['complete']
    assert read(RUN/'results/formal_complete.json')['complete']
    design=read(RUN/'configs/design.json')
    rows,ranges,overheads=[],[],[]
    for label in design['rows']:
        pid=label['policy']
        for case in design['workloads']:
            data=read(RUN/f'results/quality/{pid}/{case["label"]}.json')
            base=read(RUN/f'results/quality/uniform_dense_bf16/{case["label"]}.json')
            assert data['quality_sha256']==base['quality_sha256']==case['quality_sha256']
            differences=[a['avg_nll']-b['avg_nll'] for a,b in zip(data['blocks'],base['blocks'],strict=True)]
            assert len(differences)==64
            row={'method':label['method'],'policy':pid,'workload':case['label'],
                'batch':case['batch'],'prompt':case['prompt'],'output':case['output'],
                'delta_nll':statistics.mean(differences),'paired_se':statistics.stdev(differences)/math.sqrt(64)}
            ratios={k:[] for k in ['e2e_ms','ttft_ms','tpot_ms','cycle_ms']}
            durations={k:[] for k in ['e2e_ms','ttft_ms','tpot_ms','cycle_ms','restore_ms']}
            for campaign in range(3):
                directory=RUN/f'results/speed/round{campaign}'
                raw=read(directory/f'{pid}.json')
                timed=[s for c in raw['cases'] if c['label']==case['label'] for s in c['samples'] if not s['warmup']]
                anchors=[s for name in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']
                    for c in read(directory/f'{name}.json')['cases'] if c['label']==case['label']
                    for s in c['samples'] if not s['warmup']]
                assert len(timed)==5 and len(anchors)==10
                actual=anchors if pid=='uniform_dense_bf16' else timed
                for key in durations:durations[key].append(statistics.median(s[key] for s in actual))
                for key in ratios:ratios[key].append(statistics.median(s[key] for s in anchors)/durations[key][-1])
                overheads.append({'method':label['method'],'policy':pid,'workload':case['label'],'campaign':campaign,
                    'initialization_ms':raw['initialization_ms'],'capture_model_ms':raw['capture_model_ms'],**raw['memory']})
            row.update({key:statistics.median(values) for key,values in durations.items()})
            row.update({key.replace('_ms','_speedup'):statistics.median(values) for key,values in ratios.items()})
            rows.append(row)
            for key,values in ratios.items():
                ranges.append({'method':label['method'],'workload':case['label'],'metric':key,
                    'median_speedup':statistics.median(values),'min_campaign_speedup':min(values),'max_campaign_speedup':max(values)})
    assert len(rows)==66
    save(RUN/'results/workloads.csv',rows)
    save(RUN/'results/speed_ranges.csv',ranges)
    save(RUN/'results/process_memory.csv',overheads)
    lines=['# Qwen3-8B：固定配置的六种负载迁移（R2 Exi5 / R3 LrT2 / R4 QboE）','',
        '11行方法标签、10套唯一配置、6种负载全部保留。配置来自已封存主表，未按新负载重优化或筛选；ε是中心选择标签，不代表新负载质量保证。','',
        '同RTX5090 GPU0/NUMA0 CPU0–7、6GiB BF16 KV、native decode Graph，整批KV容量逐项检查。'
        '每策略/轮次只初始化一个引擎，依次运行六种负载，每负载3 warmup+5 timed。'
        '三轮不同顺序，BF16首尾锚点，表中归一化加速中位数；三轮范围另报，不是置信区间。'
        'TTFT取实际首token，TPOT=(E2E−TTFT)/(O−1)，cycle包含请求间恢复。','',
        '每负载在64个固定WikiText-test窗口实测教师强制decode NLL，batch与测速一致；误差为paired ±2SE的描述性范围。'
        '不同负载共用源起点以减少数据差异；该公开语料以前已经整体评测，不能声称全新未见语料。','']
    for case in design['workloads']:
        lines += [f'## {case["label"]}: B{case["batch"]}/S{case["prompt"]}/O{case["output"]}','',
            '| Method | ΔNLL | Paired ±2SE | TTFT × | TPOT × | E2E × | Cycle × | Restore ms |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:
            if row['workload']!=case['label']:continue
            lines.append(f'| {row["method"]} | {row["delta_nll"]:.5f} | {2*row["paired_se"]:.5f} | '
                f'{row["ttft_speedup"]:.3f} | {row["tpot_speedup"]:.3f} | {row["e2e_speedup"]:.3f} | '
                f'{row["cycle_speedup"]:.3f} | {row["restore_ms"]:.2f} |')
        lines.append('')
    lines += ['引擎初始化和内存峰值属于一个包含六种负载的进程，不能误报为每个单case峰值。'
        '固定O离线负载不等于连续在线服务；未据此宣称异步到达SLA保证。','',
        '[全部CSV](workloads.csv) · [三轮范围](speed_ranges.csv) · [进程内存/初始化](process_memory.csv) · [冻结设计](design.json)','']
    (RUN/'REPORT.md').write_text('\n'.join(lines).replace('](workloads.csv)','](results/workloads.csv)').replace('](speed_ranges.csv)','](results/speed_ranges.csv)').replace('](process_memory.csv)','](results/process_memory.csv)').replace('](design.json)','](configs/design.json)'))
    output=REPO/'artifacts/rebuttal/responses/reviewer2/workloads_run019'
    output.mkdir(parents=True,exist_ok=True)
    for name in ['workloads.csv','speed_ranges.csv','process_memory.csv']:shutil.copy2(RUN/'results'/name,output/name)
    shutil.copy2(RUN/'configs/design.json',output/'design.json')
    (output/'README.md').write_text('\n'.join(lines)+'\n[原始代码与结果](../../../runs/019/README.md)\n')
    write_json(RUN/'results/report_complete.json',{'complete':True,'rows':66,'policies':10,'workloads':6})
    print('Published all 66 workload rows',flush=True)


if __name__=='__main__':main()
