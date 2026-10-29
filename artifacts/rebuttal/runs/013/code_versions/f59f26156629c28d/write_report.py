"""Build all Table-2 cells from complete task records and three speed campaigns."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
TABLE = REPO / 'artifacts/rebuttal/responses/reviewer2/qwen3_table2'
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def signature(run, pid):
    policy = read(RUN.parent / run / f'configs/policies/{pid}.json')
    return hashlib.sha256(json.dumps(policy['method_map'], sort_keys=True).encode()).hexdigest()


def quality(ref, scenario):
    origin, pid = ref
    base = RUN.parent / origin
    values, evidence = {}, {}
    if scenario == 'prefill_only':
        items = [('arc_challenge', 'acc_norm,none', 'arc_c', 1172),
                 ('arc_easy', 'acc,none', 'arc_e', 2376),
                 ('winogrande', 'acc,none', 'wino', 1267), ('mmlu', 'acc,none', 'mmlu', 14042)]
        for task, metric, column, count in items:
            path = base / f'results/tasks/{pid}/{task}/full/result.json'
            data = read(path)
            assert data['limit'] is None and data['num_fewshot'] == 0
            samples = data['raw_lm_eval']['n-samples']
            assert sum(s['effective'] for s in samples.values()) == count
            assert all(s['effective'] == s['original'] for s in samples.values())
            values[column] = data['metrics'][metric] * 100
            evidence[column] = {'source': str(path.relative_to(REPO)), 'metric': metric, 'samples': count}
    else:
        for task, metric, column, count in [('cnn_dm_1000', 'rougeL_percent', 'cnn_dm', 1000),
            ('dsum', 'rougeL_percent', 'dialogsum', 1500), ('IWSLT', 'sacre_bleu', 'iwslt', 333)]:
            path = base / f'results/generation/{pid}/{task}/metrics_0000_end.json'
            data = read(path)
            assert data['samples'] == count and data['max_new_tokens'] == 256
            original_input = RUN / f'data/generation_{task}.jsonl'
            assert data['input_sha256'] == hashlib.sha256(original_input.read_bytes()).hexdigest()
            values[column] = data[metric]
            evidence[column] = {'source': str(path.relative_to(REPO)), 'metric': metric, 'samples': count}
    return values, evidence


def timing(pid, scenario, number):
    path = RUN / f'results/speed/node0/round{number}/{pid}.json'
    data = read(path)
    assert data['gpu'] == '0' and data['seed'] == 144983 and data['node'] == 0
    assert data['input_sha256'] == read(RUN / 'configs/design.json')['speed_inputs_sha256']
    assert data['first_token_observer'] == 'engine_core_before_FINAL_ONLY_filter'
    case = next(c for c in data['cases'] if c['scenario'] == scenario)
    samples = [s for s in case['samples'] if not s['warmup']]
    assert len(case['samples']) == 11 and len(samples) == 8
    assert all(s['generated_tokens'] == 8 * case['output'] for s in samples)
    assert len({s['output_sha256'] for s in samples}) == 1
    if scenario == 'prefill_only':
        metrics = {'prefill_ttft_speedup': [s['e2e_ms'] for s in samples]}
    else:
        assert all(0 < s['ttft_observed_ms'] < s['e2e_ms'] * .9 for s in samples)
        metrics = {'pd_ttft_speedup': [s['ttft_observed_ms'] for s in samples],
                   'pd_tpot_speedup': [(s['e2e_ms'] - s['ttft_observed_ms']) / 63 for s in samples],
                   'pd_e2e_speedup': [s['e2e_ms'] for s in samples]}
    return path, metrics


def main():
    assert read(RUN / 'results/worker.json')['status'] == 'complete'
    design = read(RUN / 'configs/design.json')
    for rid, digest in read(RUN / 'reference/protected_manifests.json').items():
        assert hashlib.sha256((RUN.parent / rid / 'results/manifest.json').read_bytes()).hexdigest() == digest
    for job in design['quality_jobs']:
        assert read(RUN / f'results/complete/{job["policy"]}_{job["kind"]}.json')['complete']
    repeat_rows, cache = [], {}
    for pid, spec in design['policies'].items():
        for output in spec['outputs']:
            scenario = 'prefill_only' if output == 1 else 'prefill_decode'
            rounds, paths = {}, []
            for number in range(3):
                p, metrics = timing(pid, scenario, number)
                a, anchor = timing('uniform_dense_bf16', scenario, number)
                b, end = timing('uniform_dense_bf16_bracket_end', scenario, number)
                paths.extend([str(x.relative_to(REPO)) for x in [p, a, b]])
                for column, samples in metrics.items():
                    baseline_samples = anchor[column] + end[column]
                    if pid == 'uniform_dense_bf16':
                        samples = baseline_samples
                    median = statistics.median(samples)
                    baseline = statistics.median(baseline_samples)
                    speedup = baseline / median
                    rounds.setdefault(column, []).append(speedup)
                    repeat_rows.append({'policy': pid, 'scenario': scenario, 'round': number, 'metric': column,
                        'median_ms': median, 'bf16_median_ms': baseline, 'speedup': speedup,
                        'min_timed_ms': min(samples), 'max_timed_ms': max(samples), 'timed_samples': len(samples)})
            cache[pid, scenario] = {column: {'value': statistics.median(v), 'campaign_speedups': v,
                'min_campaign_speedup': min(v), 'max_campaign_speedup': max(v), 'sources': sorted(set(paths))}
                for column, v in rounds.items()}
    columns = ['model', 'method', 'arc_c', 'arc_e', 'wino', 'mmlu', 'prefill_ttft_speedup',
               'cnn_dm', 'dialogsum', 'iwslt', 'pd_ttft_speedup', 'pd_tpot_speedup', 'pd_e2e_speedup']
    headers = ['Model', 'Method', 'ARC-C', 'ARC-E', 'Wino.', 'MMLU', 'Prefill TTFT x',
               'CNN/DM', 'DialogSum', 'IWSLT', 'P+D TTFT x', 'P+D TPOT x', 'P+D E2E x']
    rows, provenance = [], []
    for spec in design['rows']:
        row = {'model': 'Qwen3-8B', 'method': spec['method']}
        record = {'method': spec['method'], 'policies': {}, 'cells': {}}
        for scenario in ['prefill_only', 'prefill_decode']:
            entry = spec[scenario]
            assert signature(*entry['quality']) == signature('013', entry['policy'])
            values, evidence = quality(entry['quality'], scenario)
            row.update(values)
            record['cells'].update(evidence)
            record['policies'][scenario] = entry
            for column, measurement in cache[entry['policy'], scenario].items():
                row[column] = measurement['value']
                record['cells'][column] = measurement
        assert set(row) == set(columns) and all(math.isfinite(row[c]) for c in columns[2:])
        rows.append(row)
        provenance.append(record)
    for path, data, fields in [(TABLE / 'main_table.csv', rows, columns),
        (RUN / 'results/speed_repeats.csv', repeat_rows, list(repeat_rows[0]))]:
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(data)
    lines = ['# Qwen3-8B：论文 Table 2 主表', '',
        '状态：现有17行全部补齐，共187个指标。策略在本轮开始前固定；新增3组完整likelihood任务及2组完整生成任务，其余完整精度结果按相同map继承。', '',
        '速度全部来自run013：同一RTX 5090 GPU0，CPU0–7/node0，B8/S2048，Prefill O1、P+D O64，6GiB BF16 KV，eager、无prefix/chunked。三轮独立进程完整测量，每轮3 warmup+8 timed，BF16首尾锚点；主表取各轮相对本轮BF16加速比的中位数。',
        '准确率与ROUGE-L为百分数，IWSLT为SacreBLEU。ARC-C用acc_norm，ARC-E用acc；likelihood为完整zero-shot任务，生成任务CNN/DM1000、DialogSum1500、IWSLT333。', '',
        '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * 13) + ' |']
    lines += ['| ' + ' | '.join(str(row[c]) if c in columns[:2] else f'{row[c]:.2f}' for c in columns) + ' |' for row in rows]
    lines += ['', '## 速度重复范围', '',
        '以下为三轮加速比的最小值–最大值，用于展示重复测量变化，不是统计置信区间。', '',
        '| Method | Prefill TTFT x | P+D TTFT x | P+D TPOT x | P+D E2E x |', '| --- | ---: | ---: | ---: | ---: |']
    for record in provenance:
        cells = record['cells']
        lines.append('| ' + record['method'] + ' | ' + ' | '.join(
            f'{cells[c]["min_campaign_speedup"]:.2f}–{cells[c]["max_campaign_speedup"]:.2f}' for c in columns if c.endswith('_speedup')) + ' |')
    lines += ['', '## 协议与方法说明', '',
        '- Prefill TTFT采用O1完整调用时间。P+D TTFT在EngineCore输出进入FINAL_ONLY过滤前实际观测；TPOT按同一次 `(E2E−TTFT)/63` 计算，三项均有逐次原始数据。',
        '- 延续原论文计时范围：请求前的prefill权重恢复不计入E2E，另保留restore_ms与包含恢复的cycle_ms。质量评测先结束，随后执行三轮测速，未用并行质量任务期间的诊断计时填主表。',
        '- HAWQ行是Hessian＋ILP分配方法的Qwen3 PTQ移植，非原CNN integer-only/TVM全栈复现。INT4候选是Marlin W4A16；FP4低位动作在Prefill用NVFP4 W4A4，在Decode用FP4权重W4A16。',
        '- HAWQ的q为构造速度上限时配对的DASO预算标签，不代表HAWQ质量保证。INT4 q=.01的P+D原上限不可行；FP4 q=.07两场景原上限均不可行，均记录原上限并放宽至各自最小可达Linear成本，不能声称达到原目标。',
        '- FP4 q=.07的P+D最终为阶段统一配置，完整map与run001已测对照相同，生成分数继承该点；其Prefill分数本轮实际补测。',
        '- DASO + 8bit为原五候选加入SQ-INT8和FP8，共七候选。原DP行采用Prefill epsilon=.02/P+D epsilon=.01；五候选行采用两场景epsilon=.07的已有代表点，均保持原配置。',
        '- exact/validation行来自预先固定的扩展求解候选池，再按独立validation质量及校准速度筛选。epsilon=.01选择Prefill SQ-INT8/P+D BF16；epsilon=.04选择两个异构配置。该额外步骤独立于原算法，不是新kernel，也不承诺跨任务质量保证。',
        '- 旧主表、旧单轮速度及所有失败记录保留于run013/reference/table_before及旧runs；新表不按输赢选择某一轮，也未根据新下游分数调整策略。', '',
        '[主表CSV](main_table.csv) · [LaTeX](main_table.tex) · [逐项来源/哈希](provenance.json) · [逐轮速度CSV](../../../runs/013/results/speed_repeats.csv) · [run013报告](../../../runs/013/REPORT.md)', '']
    (TABLE / 'README.md').write_text('\n'.join(lines))
    tex = [r'\begin{tabular}{llrrrrrrrrrrr}', r'\hline', ' & '.join(headers) + r' \\', r'\hline']
    tex += [' & '.join(str(row[c]).replace('_', r'\_') if c in columns[:2] else f'{row[c]:.2f}' for c in columns) + r' \\' for row in rows]
    tex += [r'\hline', r'\end{tabular}', '']
    (TABLE / 'main_table.tex').write_text('\n'.join(tex))
    write(TABLE / 'provenance.json', {'status': 'complete', 'speed_campaign': '013, median of three paired campaign speedups',
        'sources_sha256': SOURCES, 'rows': provenance})
    result = {'complete': True, 'rows': 17, 'metric_cells': 187, 'missing_cells': 0,
        'unique_maps': len(design['policies']), 'unique_policy_scenarios': 32, 'speed_processes': 69,
        'new_likelihood_policy_groups': 3, 'new_generation_policy_groups': 2, 'source_files': len(SOURCES)}
    write(RUN / 'results/table_complete.json', result)
    (RUN / 'REPORT.md').write_text('# Run013：补齐Qwen3 Table 2\n\n'
        '17行187个指标全部完成，全部固定策略均保留；精度与速度来源逐项可追溯。\n\n'
        '[完整主表](../../responses/reviewer2/qwen3_table2/README.md) · [逐轮速度](results/speed_repeats.csv) · [冻结协议](configs/design.json) · [完成记录](results/table_complete.json)\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
