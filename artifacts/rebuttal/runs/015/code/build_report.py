"""Build fresh and fixed-map tables from complete quality and quiet speed data."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
TABLE = REPO / 'artifacts/rebuttal/responses/reviewer2/qwen3_table2'
COLUMNS = ['model', 'method', 'arc_c', 'arc_e', 'wino', 'mmlu', 'prefill_ttft_speedup',
           'cnn_dm', 'dialogsum', 'iwslt', 'pd_ttft_speedup', 'pd_tpot_speedup', 'pd_e2e_speedup']
HEADERS = ['Model', 'Method', 'ARC-C', 'ARC-E', 'Wino.', 'MMLU', 'Prefill TTFT x',
           'CNN/DM', 'DialogSum', 'IWSLT', 'P+D TTFT x', 'P+D TPOT x', 'P+D E2E x']


def read(path):
    return json.loads(path.read_text())


def csv_file(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows):
    return ['| ' + ' | '.join(HEADERS) + ' |', '| ' + ' | '.join(['---'] * 13) + ' |'] + [
        '| ' + ' | '.join(str(row[c]) if c in COLUMNS[:2] else f'{row[c]:.2f}' for c in COLUMNS) + ' |'
        for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-only', action='store_true')
    args = parser.parse_args()
    if args.main_only:
        assert read(RUN / 'results/priority_main_complete.json')['complete']
    else:
        assert read(RUN / 'results/worker.json')['status'] == 'complete'
    design = read(RUN / 'configs/design.json')
    for rid, digest in read(RUN / 'reference/protected_manifests.json').items():
        assert hashlib.sha256((RUN.parent / rid / 'results/manifest.json').read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location('frozen_table_readers', RUN.parent / '013/code/write_report.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.RUN = RUN
    from freeze_main import map_key
    from report_paths import record_id, anchor_id
    required = {row[s]['policy'] for row in design['rows'] for s in ['prefill_only', 'prefill_decode']}
    policies = {pid: spec for pid, spec in design['policies'].items() if not args.main_only or pid in required}
    cache, repeats, variations, overhead = {}, [], [], []
    for pid, policy in policies.items():
        for output in policy['outputs']:
            scenario = 'prefill_only' if output == 1 else 'prefill_decode'
            speeds, sources = {}, []
            for number in range(3):
                path, metrics = reference.timing(record_id(pid), scenario, number)
                _, baseline = reference.timing(anchor_id(pid), scenario, number)
                _, end = reference.timing(anchor_id(pid) + '_bracket_end', scenario, number)
                data = read(path)
                case = next(c for c in data['cases'] if c['output'] == output)
                variations.append({'policy': pid, 'scenario': scenario, 'round': number,
                    'timed_output_variants': len({s['output_sha256'] for s in case['samples'] if not s['warmup']})})
                overhead.append({'policy': pid, 'scenario': scenario, 'round': number,
                    'initialization_ms': data['initialization_ms'], 'capture_model_ms': data['capture_model_ms'],
                    'restore_ms': case['medians']['restore_ms'], 'cycle_ms': case['medians']['cycle_ms'],
                    **data['memory']})
                sources.append(str(path.relative_to(REPO)))
                for column, values in metrics.items():
                    anchor = baseline[column] + end[column]
                    if pid == 'uniform_dense_bf16':
                        values = anchor
                    value = statistics.median(values)
                    bf16 = statistics.median(anchor)
                    ratio = bf16 / value
                    speeds.setdefault(column, []).append(ratio)
                    repeats.append({'policy': pid, 'scenario': scenario, 'round': number,
                        'metric': column, 'median_ms': value, 'bf16_median_ms': bf16, 'speedup': ratio,
                        'min_timed_ms': min(values), 'max_timed_ms': max(values), 'timed_samples': len(values),
                        'sampled_other_gpu_max_utilization_percent': max(
                            (g['gpu_utilization_percent'] for s in case['samples'] if not s['warmup']
                             for at in ['gpu_before', 'gpu_after'] for g in s[at].get('other_gpus_sampled', [])), default=0)})
            cache[pid, scenario] = {c: {'value': statistics.median(x), 'campaign_speedups': x,
                'min_campaign_speedup': min(x), 'max_campaign_speedup': max(x), 'sources': sources}
                for c, x in speeds.items()}

    def table_rows(specifications):
        rows, provenance = [], []
        for spec in specifications:
            row = {'model': 'Qwen3-8B', 'method': spec['method']}
            evidence = {'method': spec['method'], 'policies': {}, 'cells': {}, 'precision_counts': {}}
            for scenario in ['prefill_only', 'prefill_decode']:
                entry = spec[scenario]
                origin, old_pid = entry['quality']
                before = read(RUN.parent / origin / f'configs/policies/{old_pid}.json')['method_map']
                actual = read(RUN / f'configs/policies/{entry["policy"]}.json')['method_map']
                evidence['precision_counts'][scenario] = dict(Counter(
                    p['prefill_method'] + ' -> ' + p['decode_method'] for p in actual.values()))
                assert map_key(before, scenario) == map_key(actual, scenario)
                quality, source = reference.quality(entry['quality'], scenario)
                row.update(quality)
                evidence['cells'].update(source)
                evidence['policies'][scenario] = entry
                for column, measured in cache[entry['policy'], scenario].items():
                    row[column] = measured['value']
                    evidence['cells'][column] = measured
            assert set(row) == set(COLUMNS) and all(math.isfinite(row[c]) for c in COLUMNS[2:])
            rows.append(row)
            provenance.append(evidence)
        return rows, provenance

    rows, provenance = table_rows(design['rows'])
    fixed, fixed_provenance = ([], []) if args.main_only else table_rows(design['fixed_rows'])
    archived = RUN / 'reference/table_before_graph'
    if not archived.exists():
        archived.mkdir(parents=True)
        for name in ['README.md', 'main_table.csv', 'main_table.tex', 'provenance.json',
                     'build_table.py', 'TPOT_DIAGNOSIS.md']:
            shutil.copy2(TABLE / name, archived / name)
    csv_file(TABLE / 'main_table.csv', rows)
    if not args.main_only:
        csv_file(RUN / 'results/fixed_main_table.csv', fixed)
    csv_file(RUN / 'results/speed_repeats.csv', repeats)
    csv_file(RUN / 'results/output_variation.csv', variations)
    csv_file(RUN / 'results/runtime_overhead.csv', overhead)
    lines = ['# Qwen3-8B：统一 Graph 运行时后的 Table 2', '',
        '17行187项指标完整。run015 在新画像上重新分配并冻结选点；相同有效map继承完整质量，新map补齐完整任务。旧表和固定旧配置的重测另行保留。', '',
        '同一RTX 5090 GPU0、NUMA0 CPU0–7，B8/S2048，Prefill O1、P+D O64，6GiB BF16 KV，无prefix/chunked。'
        '所有方法使用同一按层复用权重缓冲区、异步H2D与原生 FULL_DECODE_ONLY Graph；prefill仍为eager。'
        '三轮独立进程，每轮3 warmup+8 timed、BF16首尾锚点，表中为各轮归一化加速比的中位数。'
        '本任务不并行其他GPU质量作业；其他用户作业仅在计时外采样记录，不按这些记录筛除样本。', '',
        'ARC-C为acc_norm，ARC-E/Wino/MMLU为acc；全部zero-shot完整测试。CNN/DM1000、DialogSum1500、IWSLT333，'
        '最多256个新token、允许EOS。准确率/ROUGE-L为百分数，IWSLT为SacreBLEU。', '', *markdown(rows), '',
        '## 三轮速度范围', '', '以下是重复测量最小值–最大值，不是统计置信区间。', '',
        '| Method | Prefill TTFT x | P+D TTFT x | P+D TPOT x | P+D E2E x |', '| --- | ---: | ---: | ---: | ---: |']
    for record in provenance:
        cells = record['cells']
        lines.append('| ' + record['method'] + ' | ' + ' | '.join(
            f'{cells[c]["min_campaign_speedup"]:.2f}–{cells[c]["max_campaign_speedup"]:.2f}'
            for c in COLUMNS if c.endswith('_speedup')) + ' |')
    lines += ['', '## 解释与边界', '',
        '- 同一行的prefill和P+D使用各自固定map，与论文原表口径一致；精确policy-ID与质量来源见provenance.json。',
        '- Prefill TTFT为O1完整调用；P+D在FINAL_ONLY过滤前观测实际首token，TPOT=(同次E2E−TTFT)/63。'
        '权重恢复单列restore_ms及完整cycle_ms；表中E2E延续原协议，不含请求前恢复。',
        '- Graph依赖稳定地址。按层缓冲区取两阶段表示所需容量的较大者，保留异步搬运；实际显存、host缓存、初始化/捕获开销均另报，不能当作免费双驻留。',
        '- FP4绑定在GPU上读取原动态FP32缩放系数，使用当前stream及Graph安全workspace；量化与GEMM类型不改，24组数值对照为0误差，真实异构map多次生成与搬运权重检查通过。',
        '- 五候选DASO使用Prefill epsilon=.02/P+D epsilon=.07；七候选原DP使用Prefill epsilon=.02/P+D epsilon=.01，均按新画像重新求解。'
        '原DP网格仍是保守离散代理，exact仅指加和代理上的精确求解。',
        '- exact/validation是额外流程：候选池包括训练代理/硬件画像确定的阶段统一anchors；按独立validation mean ΔNLL+2SE+.005筛选，'
        '再取校准E2E最快点。epsilon单位为nats/token，不是准确率百分点；不承诺跨任务或分布外质量保证。'
        'validation/test为不同split；test沿用历史冻结窗口以便比较，其中部分旧map标签已在此前实验中公开，不能称为跨所有轮次从未查看过的新测试集。',
        '- HAWQ为Hessian＋ILP的Qwen3 PTQ移植，非原CNN integer-only/QAT/TVM全栈。INT4低位为Marlin W4A16；'
        'FP4低位在prefill用W4A4、decode用W4A16。HAWQ的q是配对速度上限标签，不是其NLL保证。'
        '原q配对上限不可行及最小成本放宽逐点保留在reanalysis_pool.json；主表HAWQ采用预定.80×BF16校准E2E目标，经BF16加和偏移和5%余量换算为线性上限。'
        '另报.95/.80/.65全部点；目标不构成实测E2E保证。',
        '- INT4的已知生成输出变化保留，不按token签名筛选速度。所有方法固定输入和输出长度；该协议为离线同阶段批次，未宣称任意混合在线调度支持。', '',
        '[CSV](main_table.csv) · [LaTeX](main_table.tex) · [逐项来源](provenance.json) · '
        '[固定旧配置的Graph重测](../../../runs/015/results/fixed_main_table.md) · '
        '[逐轮速度](../../../runs/015/results/speed_repeats.csv) · [开销](../../../runs/015/results/runtime_overhead.csv) · '
        '[HAWQ全部预算](../../../runs/015/results/hawq_latency_controls.csv) · '
        '[独立test约束](../../../runs/015/results/selected_test_constraints.csv) · '
        '[质量–速度图](../../../runs/015/results/heldout_frontier.pdf) · '
        '[旧eager表](../../../runs/013/results/table/README.md)', '']
    if args.main_only:
        lines = [line for line in lines if not line.startswith('[CSV]')]
        lines += ['主表17行已完成三轮完整测速；旧配置对照、额外预算及代表负载仍在后台继续，不影响本表数据完整性。', '',
            '[CSV](main_table.csv) · [LaTeX](main_table.tex) · [逐项来源](provenance.json) · '
            '[逐轮速度](../../../runs/015/results/speed_repeats.csv) · [旧eager表](../../../runs/013/results/table/README.md)', '']
    else:
        lines += ['主表组与补充组分别有各自三轮BF16首尾锚点；固定旧配置对照中的共有map沿用主表组测量，其余map使用补充组测量。', '']
    (TABLE / 'README.md').write_text('\n'.join(lines))
    if not args.main_only:
        (RUN / 'results/fixed_main_table.md').write_text('\n'.join([
            '# 固定run013的17行配置，使用统一Graph运行时重测', '',
            '配置和完整质量保持原样；使用同一GPU0及协议；共有map来自主表组，其余map来自后续补充组，分别采用各组三轮自己的BF16首尾锚点。用于区分运行时修正和重新分配的影响。', '', *markdown(fixed), '']))
    tex = [r'\begin{tabular}{llrrrrrrrrrrr}', r'\hline', ' & '.join(HEADERS) + r' \\', r'\hline']
    tex += [' & '.join(str(r[c]).replace('_', r'\_').replace('%', r'\%') if c in COLUMNS[:2]
                     else f'{r[c]:.2f}' for c in COLUMNS) + r' \\' for r in rows]
    tex += [r'\hline', r'\end{tabular}', '']
    (TABLE / 'main_table.tex').write_text('\n'.join(tex))
    (TABLE / 'provenance.json').write_text(json.dumps({'status': 'complete', 'speed_campaign': '015', 'supplemental_complete': not args.main_only,
        'sources_sha256': reference.SOURCES, 'rows': provenance}, indent=2) + '\n')
    if not args.main_only:
        (RUN / 'results/fixed_provenance.json').write_text(json.dumps(fixed_provenance, indent=2) + '\n')
    complete = {'complete': True, 'rows': len(rows), 'metric_cells': len(rows) * 11, 'missing_cells': 0,
        'supplemental_complete': not args.main_only,
        'formal_policies': len(policies), 'formal_processes': sum(len(s['order']) + 1 for s in design['rounds']
            if not args.main_only or s.get('group', 'main') == 'main'),
        'new_full_quality_jobs': len(design['quality_jobs']), 'source_files': len(reference.SOURCES)}
    (RUN / 'results/table_complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    (RUN / 'REPORT.md').write_text('# Run015：统一运行时与重新分析\n\n'
        '[完整新主表](../../responses/reviewer2/qwen3_table2/README.md) · '
        '[固定旧配置对照](results/fixed_main_table.md) · [冻结选择](configs/selected_reanalysis.json) · '
        '[延迟画像](results/graph_action_profile.csv) · [开销](results/runtime_overhead.csv)\n')
    (TABLE / 'build_table.py').write_text('''"""Rebuild the uniform-runtime table from frozen run015 evidence."""
import runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
runpy.run_path(str(ROOT / "artifacts/rebuttal/runs/015/code/build_report.py"), run_name="__main__")
'''.replace('runpy.run_path(', ("import sys\nsys.argv = [sys.argv[0], '--main-only']\n" if args.main_only else '') + 'runpy.run_path('))
    print(complete, flush=True)


if __name__ == '__main__':
    main()
