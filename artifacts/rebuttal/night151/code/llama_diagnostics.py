"""Read-only paired comparisons of the frozen original-model experiments."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT.parent/'runs'


def read(path):return json.loads(path.read_text())


def paired(source,left,right,scene):
    a=read(source/f'results/confirmation/{left}/{scene}.json')
    b=read(source/f'results/confirmation/{right}/{scene}.json')
    assert a['data_sha256']==b['data_sha256']
    assert [x['block'] for x in a['blocks']]==[x['block'] for x in b['blocks']]
    delta=[x['avg_nll']-y['avg_nll'] for x,y in zip(a['blocks'],b['blocks'],strict=True)]
    assert len(delta)==64 and all(math.isfinite(x) for x in delta)
    return statistics.mean(delta),statistics.stdev(delta)/8


def speed(source,pid,scene):
    if not (source/'results/formal_complete.json').exists():return None
    if pid not in read(source/'configs/primary_policies.json')['policies']:return None
    output=1 if scene=='prefill_only' else 64
    ratios=[]
    for number in range(3):
        folder=source/f'results/speed/node0/round{number}'
        samples=[s for c in read(folder/f'{pid}.json')['cases'] if c['output']==output
                 for s in c['samples'] if not s['warmup']]
        anchors=[s for label in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']
                 for c in read(folder/f'{label}.json')['cases'] if c['output']==output
                 for s in c['samples'] if not s['warmup']]
        assert len(samples)==8 and len(anchors)==16
        ratios.append(1. if pid=='uniform_dense_bf16' else statistics.median(s['e2e_ms'] for s in anchors)/statistics.median(s['e2e_ms'] for s in samples))
    return statistics.median(ratios)


def save(name,rows):
    with (ROOT/name).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    matched,selected,source_hashes=[],[],{}
    for rid in ['017','018']:
        source=RUNS/rid
        assert read(source/'results/confirmation_complete.json')['complete']
        pool=read(source/'configs/allocation_pool.json')
        source_hashes[rid]={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ['configs/allocation_pool.json','configs/selected_policies.json','configs/frozen_design.json']}
        points={(p['allocator'],p['scenario'],p['phase_mode'],p['latency_fraction']):p
            for p in pool['points'] if p['family'] in ['daso matched','hawq matched']}
        for key,nn in points.items():
            allocator,scene,mode,fraction=key
            if allocator!='daso':continue
            h=points['hawq_signed',scene,mode,fraction]
            assert nn['linear_limit_ms']==h['linear_limit_ms']
            assert nn['actual_e2e_target_ms']==h['actual_e2e_target_ms']
            assert nn['policy_id'] and h['policy_id']
            diff,se=paired(source,nn['policy_id'],h['policy_id'],scene)
            nmean,nse=paired(source,nn['policy_id'],'uniform_dense_bf16',scene)
            hmean,hse=paired(source,h['policy_id'],'uniform_dense_bf16',scene)
            matched.append({'run':rid,'scenario':scene,'phase_mode':mode,'latency_fraction':fraction,
                'daso_policy':nn['policy_id'],'hawq_policy':h['policy_id'],
                'linear_limit_ms':nn['linear_limit_ms'],
                'daso_predicted_linear_ms':nn['solver']['predicted_linear_ms'],
                'hawq_predicted_linear_ms':h['solver']['predicted_linear_ms'],
                'daso_delta_nll':nmean,'hawq_delta_nll':hmean,
                'paired_daso_minus_hawq':diff,'paired_two_se':2*se,
                'daso_e2e_speedup':speed(source,nn['policy_id'],scene),
                'hawq_e2e_speedup':speed(source,h['policy_id'],scene)})
        for choice in read(source/'configs/selected_policies.json')['selected']:
            mean,se=paired(source,choice['policy'],'uniform_dense_bf16',choice['scenario'])
            selected.append({'run':rid,'scenario':choice['scenario'],'budget':choice['quality_budget'],
                'selected_policy':choice['policy'],'validation_guard':choice['validation']['guarded_delta_nll'],
                'confirmation_delta_nll':mean,'confirmation_two_se':2*se,
                'mean_budget_violation':mean>choice['quality_budget'],
                'e2e_speedup':speed(source,choice['policy'],choice['scenario'])})
    save('llama_matched_allocators.csv',matched)
    save('llama_verified_budgets.csv',selected)
    (ROOT/'llama_diagnostics_sources.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    fmt=lambda x:'待测速' if x is None else f'{x:.3f}'
    lines=['# 原Llama：同候选分配器与独立预算确认（R1 AK6a / R2 Exi5 / R5 gkP9）','',
        '同候选对照使用共同BF16/INT8/FP4动作、同一实测Linear LUT、相同加和延迟上限与MILP求解规则。'
        '只替换质量目标：DASO校准代理或HAWQ signed Hessian PTQ代理。阶段统一/阶段pair分别比较；'
        '不将八候选独立阶段空间的收益全部归因于代理本身。','',
        '以下列出所有冻结的signed中心/扩展预算对照；尚未完成正式测速的点保留质量结果并明确标记。'
        '每个差值按同64窗口配对；±2SE是近似误差描述，未作多重比较校正，不据均值差宣称所有点显著优胜。','',
        '| Run | Scene | Phase mode | r | DASO ΔNLL | HAWQ ΔNLL | DASO−HAWQ ±2SE | DASO E2E × | HAWQ E2E × |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in matched:
        lines.append(f'| {row["run"]} | {row["scenario"]} | {row["phase_mode"]} | {row["latency_fraction"]:.2f} | '
            f'{row["daso_delta_nll"]:.5f} | {row["hawq_delta_nll"]:.5f} | {row["paired_daso_minus_hawq"]:.5f} ± {row["paired_two_se"]:.5f} | '
            f'{fmt(row["daso_e2e_speedup"])} | {fmt(row["hawq_e2e_speedup"])} |')
    lines += ['', '## 原池validation选择的主表预算点', '',
        '选择仅用validation的ΔNLL+2SE+0.005及校准速度，BF16保护值为0；下表报告独立test结果。'
        '这是新增DASO-8的经验保护流程，不能说原DP天然保证任意数据/任务质量；各预算共享确认样本。','',
        '| Run | Scene | ε | Selected policy | Validation guard | Test ΔNLL ±2SE | E2E × | Mean exceeds ε |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: | --- |']
    for row in selected:
        lines.append(f'| {row["run"]} | {row["scenario"]} | {row["budget"]:.2f} | {row["selected_policy"]} | '
            f'{row["validation_guard"]:.5f} | {row["confirmation_delta_nll"]:.5f} ± {row["confirmation_two_se"]:.5f} | '
            f'{fmt(row["e2e_speedup"])} | {row["mean_budget_violation"]} |')
    lines += ['', '质量预算针对WikiText的教师强制NLL；完整生成与MC另测，不能用本表代替。'
        '速度为三轮各自BF16首尾锚点归一化后中位数，均在GPU0/NUMA0；非核心点未借用calibration数值冒充正式速度。','',
        '[同候选逐点CSV](llama_matched_allocators.csv) · [预算CSV](llama_verified_budgets.csv) · '
        '[冻结来源SHA](llama_diagnostics_sources.json)','']
    (ROOT/'LLAMA_DIAGNOSTICS.md').write_text('\n'.join(lines))
    print('Matched',len(matched),'selected',len(selected),'budget violations',sum(r['mean_budget_violation'] for r in selected))


if __name__=='__main__':main()
