"""Publish every matched point, then add preregistered complete task metrics."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

from common import RUN, REPO, write_json
from launch import read

OUTPUT = REPO / 'artifacts/rebuttal/responses/reviewer2/matched_allocator_run016'


def nll(pid, scenario, split='confirmation'):
    value = read(RUN / f'results/{split}/{pid}/{scenario}.json')
    baseline = read(RUN / f'results/{split}/uniform_dense_bf16/{scenario}.json')
    design = read(RUN / 'configs/frozen_design.json')
    assert value['data_sha256'] == design['datasets'][split]['sha256']
    differences = [a['avg_nll']-b['avg_nll'] for a,b in zip(value['blocks'],baseline['blocks'],strict=True)]
    assert len(differences) == design['datasets'][split]['blocks']
    return statistics.mean(differences), statistics.stdev(differences)/math.sqrt(len(differences))


def speed(pid, scenario):
    output = 1 if scenario == 'prefill_only' else 64
    records, ratios, durations, restores, cycles = [], [], [], [], []
    for number in range(3):
        path = RUN / f'results/speed/node0/round{number}/{pid}.json'
        data = read(path)
        assert data['mode'] == 'arena_graph' and data['gpu'] == '0' and data['seed'] == 144983
        case = next(c for c in data['cases'] if c['output'] == output)
        timed = [s for s in case['samples'] if not s['warmup']]
        assert len(case['samples']) == 11 and len(timed) == 8
        assert all(s['generated_tokens'] == 8*output for s in timed)
        anchors = []
        for suffix in ['', '_bracket_end']:
            bf16 = read(RUN / f'results/speed/node0/round{number}/uniform_dense_bf16{suffix}.json')
            anchors += [s for c in bf16['cases'] if c['output'] == output for s in c['samples'] if not s['warmup']]
        assert len(anchors) == 16
        sample = anchors if pid == 'uniform_dense_bf16' else timed
        latency = statistics.median(s['e2e_ms'] for s in sample)
        ratios.append(statistics.median(s['e2e_ms'] for s in anchors)/latency)
        durations.append(latency)
        restores.append(statistics.median(s['restore_ms'] for s in sample))
        cycles.append(statistics.median(s['cycle_ms'] for s in sample))
        records.extend(timed)
    return {'e2e_ms': statistics.median(durations), 'speedup': statistics.median(ratios),
        'min_campaign_speedup': min(ratios), 'max_campaign_speedup': max(ratios),
        'restore_ms': statistics.median(restores), 'cycle_ms': statistics.median(cycles),
        'samples': records}


def save_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_rows(pool):
    cache, rows = {}, []
    for point in pool['points']:
        pid, scenario = point['policy_id'], point['scenario']
        row = {'label': point['label'], 'policy': pid, 'scenario': scenario, 'family': point['family'],
            'phase_mode': point.get('phase_mode'), 'latency_fraction': point.get('latency_fraction'),
            'linear_limit_ms': point.get('linear_limit_ms'), 'e2e_target_ms': point.get('actual_e2e_target_ms'),
            'quality_budget': point.get('quality_budget'), 'status': 'infeasible' if pid is None else 'complete',
            'confirmation_delta_nll': None, 'paired_se': None, 'e2e_ms': None, 'speedup': None,
            'min_campaign_speedup': None, 'max_campaign_speedup': None, 'restore_ms': None, 'cycle_ms': None,
            'e2e_target_violations': None, 'cycle_target_violations': None}
        if pid:
            key = pid, scenario
            if key not in cache:
                mean, se = nll(pid, scenario)
                cache[key] = mean, se, speed(pid, scenario)
            mean, se, timing = cache[key]
            row.update(confirmation_delta_nll=mean, paired_se=se,
                       **{k:v for k,v in timing.items() if k != 'samples'})
            cap = point.get('actual_e2e_target_ms')
            if cap:
                row['e2e_target_violations'] = sum(s['e2e_ms'] > cap for s in timing['samples'])
                row['cycle_target_violations'] = sum(s['cycle_ms'] > cap for s in timing['samples'])
        rows.append(row)
    return rows, cache


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    cases = [('prefill_only', 'same_operator', 'Prefill'),
             ('prefill_decode', 'same_operator', 'P+D: same operator'),
             ('prefill_decode', 'phase_pair', 'P+D: phase-specific backend')]
    for axis, (scenario, mode, title) in zip(axes, cases, strict=True):
        for family, color, marker in [('daso matched','#009E73','s'),('hawq matched','#0072B2','o')]:
            local = [r for r in rows if r['family'] == family and r['scenario'] == scenario and
                     r['phase_mode'] == mode and r['status'] == 'complete']
            axis.errorbar([r['confirmation_delta_nll'] for r in local], [r['speedup'] for r in local],
                xerr=[2*r['paired_se'] for r in local],
                yerr=[[r['speedup']-r['min_campaign_speedup'] for r in local],
                      [r['max_campaign_speedup']-r['speedup'] for r in local]],
                fmt=marker, color=color, capsize=2, label=family)
        axis.set(title=title, xlabel='Confirmation delta NLL (nats/token)', ylabel='E2E speedup vs BF16')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for suffix in ['pdf','svg','png']:
        fig.savefig(RUN / f'results/matched_frontier.{suffix}', dpi=180)
    plt.close(fig)


def full_tasks():
    spec = importlib.util.spec_from_file_location('frozen_quality_reader', RUN.parent / '013/code/write_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN = RUN
    rows = []
    for item in read(RUN / 'configs/full_quality_targets.json'):
        metrics, sources = module.quality(['016',item['policy']],item['scenario'])
        rows.append({'label': item['label'], 'policy': item['policy'], 'scenario': item['scenario'],
                     'metrics': metrics, 'sources': sources})
    write_json(RUN / 'results/full_task_metrics.json', rows)
    flattened = []
    for item in rows:
        for metric,value in item['metrics'].items():
            flattened.append({'label':item['label'],'policy':item['policy'],'scenario':item['scenario'],
                              'metric':metric,'value':value,'source':item['sources'][metric]['source']})
    save_csv(RUN / 'results/full_task_metrics.csv', flattened)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    assert read(RUN / 'results/formal_complete.json')['complete']
    pool = read(RUN / 'configs/allocation_pool.json')
    selection = read(RUN / 'configs/selected_policies.json')
    assert selection['pool_sha256'] == hashlib.sha256((RUN / 'configs/allocation_pool.json').read_bytes()).hexdigest()
    rows, cache = point_rows(pool)
    save_csv(RUN / 'results/allocation_points.csv', rows)
    selected = []
    for p in selection['selected']:
        mean, se = nll(p['policy'], p['scenario'])
        timing = speed(p['policy'],p['scenario'])
        selected.append({'label':p['label'],'policy':p['policy'],'scenario':p['scenario'],
            'quality_budget':p['quality_budget'],'confirmation_delta_nll':mean,'paired_se':se,
            'mean_budget_violation':mean>p['quality_budget'],
            **{k:v for k,v in timing.items() if k != 'samples'}})
    save_csv(RUN / 'results/selected_quality_points.csv', selected)
    plot(rows)
    lines = ['# Qwen3同候选分配器与阶段适配对照', '',
        '三轮同卡速度与新confirmation窗口结果已完成。' + ('预定完整任务也已完成。' if args.full else '预定完整MC/生成任务仍在继续，以下先报告NLL与速度。'), '',
        '所有主对照共用BF16、plain INT8、NVFP4，三种延迟目标及同一MILP。'
        '两者分别最小化DASO质量代理或Hessian加权权重误差；并列解用同样的最小预测延迟规则。'
        'HAWQ为Hessian/ILP的PTQ移植，不称为原CNN QAT/TVM全栈复现。'
        'prefill/decode同算子与阶段后端适配均在评估前固定，现有run015强对照继续保留。', '',
        'validation32和confirmation64与历史对应64窗口不重叠，但benchmark以前已有整体评估，不能称为全新语料。'
        '主负载B8/S2048/O1或64，GPU0/NUMA0 CPU0–7、6GiB BF16 KV、native decode Graph；三轮3warmup+8timed、BF16首尾锚点。'
        '表中为三轮归一化加速比中位数，范围为重复最小/最大值，不是置信区间。图中横线为窗口paired ±2SE的描述性误差条。恢复另列cycle。', '',
        '## 同候选与同预测延迟上限', '',
        '| Scenario / mode | Allocator | Target / BF16 | Confirmation ΔNLL | Paired ±2SE | E2E × | Campaign range | Target misses /24 |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in rows:
        if r['family'] not in ['daso matched','hawq matched','daso independent matched formats']:
            continue
        if r['status'] != 'complete':
            lines.append(f'| {r["scenario"]}/{r["phase_mode"]} | {r["family"]} | {r["latency_fraction"]} | infeasible | — | — | — | — |')
        else:
            lines.append(f'| {r["scenario"]}/{r["phase_mode"] or "independent"} | {r["family"]} | {r["latency_fraction"]:.2f} | '
                f'{r["confirmation_delta_nll"]:.5f} | {2*r["paired_se"]:.5f} | {r["speedup"]:.3f} | '
                f'{r["min_campaign_speedup"]:.3f}–{r["max_campaign_speedup"]:.3f} | {r["e2e_target_violations"]} |')
    lines += ['', '## 八候选扩展的validation选择', '',
        '该部分包含新增plain INT8训练与候选池校验，不能单独归因为分配器优于HAWQ。', '',
        '| Scenario | ε | Confirmation ΔNLL | Mean violation | E2E × | Policy |',
        '| --- | ---: | ---: | --- | ---: | --- |']
    for r in selected:
        lines.append(f'| {r["scenario"]} | {r["quality_budget"]:.2f} | {r["confirmation_delta_nll"]:.5f} | '
                     f'{r["mean_budget_violation"]} | {r["speedup"]:.3f} | {r["policy"]} |')
    if args.full:
        from preparation_costs import main as preparation
        preparation()
        assert read(RUN / 'results/full_quality_complete.json')['complete']
        complete = full_tasks()
        lines += ['', '## 预定完整任务指标', '',
            '中心.80对照与两个质量预算选择点全部列出，不按分数选择展示对象。Prefill为完整zero-shot ARC-C/ARC-E/Wino/MMLU；'
            '生成CNN/DM1000、DialogSum1500、IWSLT333，native Qwen3 non-thinking、greedy、最多256新token允许EOS。', '',
            '| Label | Scenario | Metrics |', '| --- | --- | --- |']
        lines += [f'| {x["label"]} | {x["scenario"]} | ' + ', '.join(f'{k}={v:.2f}' for k,v in x['metrics'].items()) + ' |' for x in complete]
    lines += ['', '## 解释边界', '',
        '- 同一预测Linear上限不等于相同实测E2E；目标违约、恢复cycle和真实质量都保留，不能只比较名义r或ε。',
        '- 原始signed Hessian为主对照；clip_zero/absolute只在预定中心预算作敏感性控制。硬件最小成本端点单列，并非Hessian模型胜出的证据。',
        '- 共同使用实际融合权重的量化扰动，NV4/W4共享canonical表示；DASO第八动作有独立采集的误差/训练标签，没有复制SQ系数。',
        '- 质量模型结构和超参数沿用原实现；本实验共用MILP，不能据此声称原DP离散实现已被证明最优。',
        '- 计算成本包含新增训练标签和validation搜索；旧Hessian/原标签属于有成本的缓存复用，不能当成从零免费。', '',
        '[全部预登记点](results/allocation_points.csv) · [选择点](results/selected_quality_points.csv) · '
        '[质量–速度图](results/matched_frontier.pdf) · [冻结设计](configs/frozen_design.json) · '
        '[全部原始代码与日志](../../../runs/016/README.md)', '']
    (RUN / 'REPORT.md').write_text('\n'.join(lines).replace('(../../../runs/016/README.md)', '(README.md)'))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    import shutil
    for folder in ['results','configs']:
        (OUTPUT/folder).mkdir(exist_ok=True)
    for name in ['allocation_points.csv','selected_quality_points.csv','matched_frontier.pdf']:
        shutil.copy2(RUN/'results'/name,OUTPUT/'results'/name)
    if args.full:
        for name in ['full_task_metrics.csv','preparation_summary.csv','preparation_processes.csv','preparation_scope.json']:
            shutil.copy2(RUN/'results'/name,OUTPUT/'results'/name)
    for name in ['frozen_design.json','allocation_pool.json','selected_policies.json']:
        shutil.copy2(RUN/'configs'/name,OUTPUT/'configs'/name)
    (OUTPUT/'README.md').write_text('\n'.join(lines))
    write_json(RUN/'results/report_complete.json',{'complete':True,'full_tasks_complete':args.full,
        'labels':len(rows),'unique_policy_scenarios':len(cache),'selected_points':len(selected),
        'confirmation_budget_violations':sum(r['mean_budget_violation'] for r in selected)})
    print('Published matched comparison; full tasks:',args.full,flush=True)


if __name__ == '__main__':
    main()
