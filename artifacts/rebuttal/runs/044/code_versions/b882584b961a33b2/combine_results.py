"""Deliver all budget/weight points together, with per-campaign speed sources."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parents[2]
DEST = RUNS.parent / 'responses/reviewer2/qwen3_budget_prefill_weight'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: list(csv.DictReader(p.open()))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    provenance, rows, ranges, distributions = {}, [], {}, {}
    for rid in ['032', '033']:
        root = RUNS / rid
        audit = read(root / 'results/final_audit.json')
        assert audit['complete'] and audit['verified']
        provenance[rid] = {'audit_sha256': sha(root / 'results/final_audit.json'),
            'report_sha256': sha(root / 'REPORT.md'), 'csv_sha256': audit['csv_sha256'],
            'formal_device': audit.get('formal_device', {'physical_gpu':0})}
        for name, digest in audit['csv_sha256'].items():
            assert sha(root / 'results' / name) == digest
        for point in load(root / 'results/formal_campaigns.csv'):
            ranges.setdefault((rid, point['policy']), []).append(float(point['speedup_int8']))
        distributions[rid] = read(root / 'results/policy_counts.json')
        for row in load(root / 'results/sweep_results.csv'):
            rows.append({**row, 'campaign': rid})
        if rid == '032':
            speed = {r['policy']: r for r in load(root / 'results/formal_speed.csv')}
            gen = {r['policy']: r for r in load(root / 'results/full_generation.csv')}
            mc = {r['policy']: r for r in load(root / 'results/full_mc.csv')}
            nll = {r['policy']: r for r in load(root / 'results/benchmark_nll.csv')}
            counts = read(root / 'results/policy_counts.json')
            controls = [{'label': p, 'policy': p, 'prefill_weight': '', 'budget': '',
                'weighted_prediction': '', 'original_prediction': '',
                'sparse_fp4_prefill': counts[p]['prefill'].get('sparse_nvfp4', 0),
                **nll[p], **speed[p], **{k:v for k,v in gen[p].items() if k not in ['policy','source']},
                **{k:v for k,v in mc[p].items() if k!='policy'}, 'campaign': rid}
                for p in read(root / 'configs/design.json')['anchors']]
    rows = controls + rows
    fields = ['label','policy','campaign','prefill_weight','budget','sparse_fp4_prefill',
        'prefill_sq_int8','prefill_int8','prefill_dense_fp4','prefill_sparse_fp4','prefill_bf16','decode_w4','decode_sq_int8',
        'arc_c','arc_e','wino','mmlu','mc_avg','cnn_dm','dialogsum','iwslt','generation_avg',
        'ttft_ms','tpot_ms','e2e_ms','speedup_int8','speedup_int8_min','speedup_int8_max','speedup_bf16','restore_ms','cycle_ms',
        'weighted_prediction','original_prediction','delta_nll','paired_se','guarded_delta_nll',
        'mean_within_nominal_budget','guard_within_nominal_budget','prefill_source']
    baseline = next(r for r in rows if r['label']=='uniform_int8')
    for r in rows:
        distribution = distributions[r['campaign']][r['policy']]
        assert all(sum(distribution[p].values()) == 144 for p in ['prefill','decode'])
        for target, phase, method in [('prefill_sq_int8','prefill','sq_int8'),('prefill_int8','prefill','int8'),
            ('prefill_dense_fp4','prefill','dense_nvfp4'),('prefill_sparse_fp4','prefill','sparse_nvfp4'),
            ('prefill_bf16','prefill','dense_bf16'),('decode_w4','decode','w4a16_ours'),('decode_sq_int8','decode','sq_int8')]:
            r[target] = distribution[phase].get(method,0)
        values = ranges[r['campaign'], r['policy']]
        r['speedup_int8_min'], r['speedup_int8_max'] = min(values), max(values)
        assert abs(float(r['mc_avg'])-statistics.mean(float(r[k]) for k in ['arc_c','arc_e','wino','mmlu'])) < 1e-6
        assert abs(float(r['generation_avg'])-statistics.mean(float(r[k]) for k in ['cnn_dm','dialogsum','iwslt'])) < 1e-6
    with (DEST / 'complete_comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    labels = {'uniform_dense_bf16':'BF16','uniform_int8':'INT8','uniform_sq_int8':'SQ-INT8',
        'phase_sq_int8_w4a16_ours':'SQ prefill / full W4 decode','group_7fff':'029 fixed-SQ / 135 W4'}
    title = lambda r: labels.get(r['label'], r['label'])
    lines = ['# Qwen3-8B：完整DASO P+D预算与prefill加权对照（Reviewer2）', '',
        '原始run016主表保留。本表对比同一Qwen八候选代理下的完整空间预算/权重配置，所有模块与阶段格式自由选择。032原六标签在任务前冻结；033的λ32/128是看到部分预算曲线后、依据CPU阶段代价尺度设计的独立补充，并在其自身GPU输出前冻结。全部点保留，不称为新盲测或通用最优参数。', '',
        'λ2/q=.03与λ1/q=.02是同一物理配置，不算独立结果。q为代理/加权偏好预算，实际NLL另列；λ≠1不是重新校准的ΔNLL。MC Avg和生成Avg是各自算术摘要，不把两个Avg或全部任务混成一个准确率。', '',
        '| 设置 | ARC-C | ARC-E | Wino | MMLU | MC Avg | CNN/DM | DialogSum | IWSLT | Gen Avg | P+D TTFT ms | TPOT ms | E2E ×INT8 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        keys = ['arc_c','arc_e','wino','mmlu','mc_avg','cnn_dm','dialogsum','iwslt','generation_avg','ttft_ms','tpot_ms','speedup_int8']
        lines.append('| ' + title(r) + ' | ' + ' | '.join(f'{float(r[k]):.3f}' for k in keys) + ' |')
    device_note = '；'.join(f'{rid}使用GPU{p["formal_device"]["physical_gpu"]}' for rid,p in provenance.items())
    migration_note = ('033因GPU0资源等待迁卡，完整三轮及对照统一重测；旧GPU0未完成组记录保留作诊断，不混入正式平均。'
        if 'archived_gpu0_records' in provenance['033']['formal_device'] else '')
    lines += ['', f'P+D速度：{device_note}，各campaign内统一同卡及对应NUMA、B8/S2048/O64、native decode Graph/6GiB KV、三轮3warmup+5timed。每点相对其所在campaign的同轮INT8归一化，CSV保留032/033来源；没有额外测prefill-only速度，表中不把P+D TTFT冒充该列。恢复和cycle另存，不计入论文单批E2E。'+migration_note, '',
        'MC四任务完整0-shot/batch4/eager/3GiB KV，同prefill只计算一次，关闭未使用decode搬运。MC对应所列P+D配置自身的prefill，不是另行优化的prefill-only策略；原run016主表两场景分别选点，需要区分。完整生成1000/1500/333；相同map/输入/代码协议的旧Graph结果按SHA复用，详见各run来源。全部参数的任务分数均展示，不据最终任务分数回填训练或重新选择参数。', '',
        '## 原预测与实际NLL', '',
        '| 设置 | 加权预测 | 原未加权预测 | 实测ΔNLL | 实测保护值 | 名义预算 | 均值通过 | 保护值通过 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    fmt = lambda v: f'{float(v):.5f}' if str(v) not in ['', 'None'] else '—'
    for r in rows:
        lines.append('| '+title(r)+' | '+' | '.join(fmt(r.get(k,'')) for k in
            ['weighted_prediction','original_prediction','delta_nll','guarded_delta_nll','budget'])+
            ' | '+str(r.get('mean_within_nominal_budget','—'))+' | '+str(r.get('guard_within_nominal_budget','—'))+' |')
    lines += ['', 'NLL使用已经评估过的WikiText test固定100窗口，保护值=mean+2SE+.005，BF16为0。实际生成/MC分数高，不代表NLL预算必然通过；均值与保护值分开报告。没有bootstrap显著性结论，也没有在本轮重测HAWQ。', '',
        '## 自动精度分配', '',
        '| 设置 | P SQ-INT8 | P INT8 | P稠密FP4 | P稀疏FP4 | D W4A16 | D SQ-INT8 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if r['prefill_weight']:
            lines.append('| '+title(r)+' | '+' | '.join(str(r[k]) for k in
                ['prefill_sq_int8','prefill_int8','prefill_dense_fp4','prefill_sparse_fp4','decode_w4','decode_sq_int8'])+' |')
    by_label = {r['label']:r for r in rows}
    tight, loose, sq = [by_label[k] for k in ['w1_e0.01','w1_e0.04','uniform_sq_int8']]
    lines += ['', '## 本轮观察', '',
        f'原权重下，预算从.04收紧至.01，稀疏FP4 prefill从{loose["prefill_sparse_fp4"]}减至{tight["prefill_sparse_fp4"]}个，'
        f'生成Avg从{float(loose["generation_avg"]):.3f}升至{float(tight["generation_avg"]):.3f}，'
        f'MC Avg从{float(loose["mc_avg"]):.3f}升至{float(tight["mc_avg"]):.3f}。'
        '小权重主要沿收紧预算的方向移动，λ2/.03与原权重.02完全重合。', '']
    for label in ['w32_e0.03','w128_e0.03']:
        r = by_label[label]
        lines.append(f'{label}：生成Avg {float(r["generation_avg"]):.3f}（相对INT8 {float(r["generation_avg"])-float(baseline["generation_avg"]):+.3f}），'
            f'MC Avg {float(r["mc_avg"]):.3f}（相对INT8 {float(r["mc_avg"])-float(baseline["mc_avg"]):+.3f}、相对SQ-INT8 {float(r["mc_avg"])-float(sq["mc_avg"]):+.3f}个百分点），'
            f'E2E为同轮INT8的{float(r["speedup_int8"]):.3f}倍速度（三轮范围{r["speedup_int8_min"]:.3f}–{r["speedup_int8_max"]:.3f}）。'
            f'实际NLL保护值{float(r["guarded_delta_nll"]):.5f}，名义加权预算.03。')
        lines.append('')
    lines += ['这些点表明提高prefill权重能在完整自由搜索中改变分配、改善生成质量折中；仍不能宣称所有任务全面超过8bit，或实测NLL保护值满足.03。两个大权重点的decode也都全W4，在这一固定decode子空间内同样可以重参数化成原预算；本轮没有证明它们优于更密的原预算曲线。λ32/128属于探索性阶段权重敏感性结果，不能仅据本表宣布通用默认参数；原代理阶段尺度和真实任务质量仍有差距。', '']
    lines += ['',
        '[完整CSV](complete_comparison.csv) · [精度—速度图](quality_speed.png) · [PDF](quality_speed.pdf) · [032原始报告](../../../runs/032/REPORT.md) · [033补充报告](../../../runs/033/REPORT.md) · [来源哈希](provenance.json)', '']
    (DEST / 'README.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ytitle in zip(axes, ['generation_avg','mc_avg'], ['Generation average score','Prefill MC average accuracy (%)']):
        ax.axhline(float(baseline[key]), color='gray', linestyle='--', linewidth=.9, label='INT8 quality')
        ax.axvline(1., color='gray', linestyle=':', linewidth=.9)
        for r in rows:
            if r['label']=='w2_e0.03':
                continue
            weight = float(r['prefill_weight']) if r['prefill_weight'] else None
            marker, color = ('o','#2469a0') if weight==1 else ('D','#a44e16') if weight else ('s','#555555')
            short = {'phase_sq_int8_w4a16_ours':'SQ/W4','group_7fff':'SQ/135W4'}
            label = ('q='+str(r['budget'])) if weight==1 else ('λ='+f'{weight:g}') if weight else short.get(r['label'],title(r))
            x,y=float(r['speedup_int8']),float(r[key])
            ax.errorbar(x,y,xerr=[[x-r['speedup_int8_min']],[r['speedup_int8_max']-x]],
                fmt='none',ecolor=color,alpha=.45,capsize=2,linewidth=.8)
            ax.scatter(x,y,marker=marker,color=color,s=38,zorder=3)
            offset = {'uniform_sq_int8':(-38,-13),'group_7fff':(-44,-14)}.get(r['label'],(4,5))
            ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=7)
        ax.set_xlabel('Single-batch E2E speedup over same-campaign INT8')
        ax.set_ylabel(ytitle)
        ax.grid(alpha=.2)
    fig.suptitle('Qwen3-8B full-space DASO: budget and prefill-weight sensitivity',fontsize=12)
    fig.tight_layout()
    fig.savefig(DEST/'quality_speed.png',dpi=200,bbox_inches='tight')
    fig.savefig(DEST/'quality_speed.pdf',bbox_inches='tight')
    plt.close(fig)
    provenance['delivery_sha256'] = {p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='provenance.json'}
    (DEST/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Delivered',len(rows),'rows to',DEST,flush=True)


if __name__ == '__main__':
    main()
