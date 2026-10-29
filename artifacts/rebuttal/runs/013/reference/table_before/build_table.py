"""Render the available Table-2 evidence; unmeasured cells remain empty."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RUNS = ROOT / 'artifacts/rebuttal/runs'
sources = {}


def read(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def signature(run, policy):
    data = read(RUNS / run / f'configs/policies/{policy}.json')
    return hashlib.sha256(json.dumps(data['method_map'], sort_keys=True).encode()).hexdigest()


def quality(run, policy, scenario):
    values, evidence = {}, {}
    tasks = [('arc_challenge', 'acc_norm,none', 'arc_c'),
             ('arc_easy', 'acc,none', 'arc_e'),
             ('winogrande', 'acc,none', 'wino'), ('mmlu', 'acc,none', 'mmlu')]
    generation = [('cnn_dm_1000', 'rougeL_percent', 'cnn_dm', 1000),
                  ('dsum', 'rougeL_percent', 'dialogsum', 1500),
                  ('IWSLT', 'sacre_bleu', 'iwslt', 333)]
    if scenario == 'prefill_only':
        for task, metric, column in tasks:
            path = RUNS / run / f'results/tasks/{policy}/{task}/full/result.json'
            data = read(path)
            assert data['limit'] is None
            values[column] = data['metrics'][metric] * 100
            evidence[column] = {'source': str(path.relative_to(ROOT)), 'metric': metric}
    else:
        for task, metric, column, count in generation:
            path = RUNS / run / f'results/generation/{policy}/{task}/metrics_0000_end.json'
            data = read(path)
            assert data['samples'] == count
            values[column] = data[metric]
            evidence[column] = {'source': str(path.relative_to(ROOT)), 'metric': metric}
    return values, evidence


summary_path = RUNS / '011/results/summary_final.csv'
sources[str(summary_path.relative_to(ROOT))] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
speed_rows = {(r['label'], r['scenario']): r for r in csv.DictReader(summary_path.open())}


def speed(label, scenario):
    row = speed_rows[label, scenario]
    policy = row['policy_id']
    folder = RUNS / '011/results/calibration_speed_quiet'
    anchors = [read(folder / pid / f'{scenario}.json') for pid in
               ['uniform_dense_bf16', 'uniform_dense_bf16_bracket_end']]
    bf16 = statistics.median(x['elapsed_ms'] for d in anchors for x in d['iterations'] if not x['warmup'])
    path = folder / policy / f'{scenario}.json'
    data = read(path)
    assert data['gpu'] == '0' and data['batch'] == 8 and data['input'] == 2048
    assert data['output'] == (1 if scenario == 'prefill_only' else 64)
    elapsed = bf16 if policy == 'uniform_dense_bf16' else statistics.median(
        x['elapsed_ms'] for x in data['iterations'] if not x['warmup'])
    result = bf16 / elapsed
    assert abs(result - float(row['speedup'])) < 1e-10
    return policy, result, {'source': str(path.relative_to(ROOT)),
                            'bf16_anchors': [str((folder / p / f'{scenario}.json').relative_to(ROOT))
                                             for p in ['uniform_dense_bf16', 'uniform_dense_bf16_bracket_end']],
                            'initial_campaign_speedup': float(row['initial_speedup'])}


# These are existing named representatives, not newly chosen task-score maxima.
# P and P+D policies are separate, as in the paper's Table 2.
specs = []
for name, policy in [('BF16', 'dense_bf16'), ('Dense NVFP4', 'dense_nvfp4'),
                     ('W4A16 NVFP4', 'w4a16_ours'), ('W4A16 INT4 Marlin', 'marlin_int4'),
                     ('Sparse BF16', 'sparse_bf16'), ('Sparse NVFP4', 'sparse_nvfp4'),
                     ('INT8 W8A8', 'int8'), ('SmoothQuant INT8', 'sq_int8'),
                     ('FP8 W8A8', 'fp8'), ('SmoothQuant FP8', 'sq_fp8')]:
    pid = 'uniform_' + policy
    current = pid if (pid, 'prefill_only') in speed_rows else None
    specs.append((name, ('002', pid), ('002', pid), current, current))
specs += [
    ('HAWQ-v3 INT4, q=.01 (PTQ adaptation)', None, None,
     'hawq_marlin_int4_signed_prefill_only_q0.01', 'hawq_marlin_int4_signed_prefill_decode_q0.01'),
    ('HAWQ-v3 FP4, q=.01 (PTQ adaptation)', None, None,
     'hawq_fp4_phase_pair_signed_prefill_only_q0.01', 'hawq_fp4_phase_pair_signed_prefill_decode_q0.01'),
    ('HAWQ FP4 minimum-cost control, q=.07', None, ('001', 'phase_uniform_nvfp4_marlin'),
     'hawq_fp4_phase_pair_signed_prefill_only_q0.07', 'hawq_fp4_phase_pair_signed_prefill_decode_q0.07'),
    ('DASO, original 5 candidates (representatives)', ('002', 'daso5_measured_prefill_only_02'),
     ('002', 'daso5_measured_prefill_decode_04'), None, 'transfer_daso5_measured_prefill_decode_04'),
    ('DASO + 8bit, original DP (representatives)', ('002', 'daso8_prefill_only_02'),
     ('002', 'daso8_prefill_decode_01'), 'transfer_daso8_prefill_only_02', 'transfer_daso8_prefill_decode_01'),
    ('DASO + 8bit + exact/validation, epsilon=.01', ('010', 'verified_c63754e060c4'),
     ('010', 'verified_7aafcc63be76'), 'verified_prefill_only_q0.01', 'verified_prefill_decode_q0.01'),
    ('DASO + 8bit + exact/validation, epsilon=.04', ('010', 'verified_d1eeaeaf84b5'),
     ('010', 'verified_9e60153b6827'), 'verified_prefill_only_q0.04', 'verified_prefill_decode_q0.04'),
]
columns = ['model', 'method', 'arc_c', 'arc_e', 'wino', 'mmlu', 'prefill_ttft_speedup',
           'cnn_dm', 'dialogsum', 'iwslt', 'pd_ttft_speedup', 'pd_tpot_speedup', 'pd_e2e_speedup']
headers = ['Model', 'Method', 'ARC-C', 'ARC-E', 'Wino.', 'MMLU', 'Prefill TTFT x',
           'CNN/DM', 'DialogSum', 'IWSLT', 'P+D TTFT x', 'P+D TPOT x', 'P+D E2E x']
rows, provenance = [], []
for name, p_quality, d_quality, p_label, d_label in specs:
    row = dict.fromkeys(columns, '')
    row.update(model='Qwen3-8B', method=name)
    record = {'method': name, 'cells': {}, 'policies': {}}
    for scenario, quality_ref, label, column in [
        ('prefill_only', p_quality, p_label, 'prefill_ttft_speedup'),
        ('prefill_decode', d_quality, d_label, 'pd_e2e_speedup')]:
        if quality_ref:
            values, evidence = quality(*quality_ref, scenario)
            row.update(values)
            record['cells'].update(evidence)
            record['policies'][scenario + '_quality'] = list(quality_ref)
        if label:
            policy, value, evidence = speed(label, scenario)
            row[column] = value
            record['cells'][column] = evidence
            record['policies'][scenario + '_speed'] = ['011', policy]
            if quality_ref:
                assert signature(*quality_ref) == signature('011', policy), (name, scenario)
    if name == 'BF16':
        row.update(pd_ttft_speedup=1.0, pd_tpot_speedup=1.0)
        for column in ['pd_ttft_speedup', 'pd_tpot_speedup']:
            record['cells'][column] = {'derivation': 'BF16 divided by itself; identity, not a new timing measurement'}
    record['missing'] = [c for c in columns[2:] if row[c] == '']
    rows.append(row)
    provenance.append(record)


def display(value):
    return f'{value:.2f}' if isinstance(value, (int, float)) else (value or '待补')


with (HERE / 'main_table.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, columns)
    writer.writeheader()
    writer.writerows(rows)
report = ['# Qwen3-8B：论文 Table 2 格式的现有证据汇总', '',
          '状态：尚未全部填满，不能称为已完成的最终主表。此表只整理已有记录，不启动新实验、不修改已封存 runs。', '',
          '速度统一取 run011 最后一轮 GPU0、B8/S2048、O1/O64、6GiB KV 的结果。未在该轮测量的速度留空，不混入 run001/002/003 的数字。',
          '精度来自完整下游评测原始 JSON；连接精度与速度前核对完整 method-map。准确率与 ROUGE-L 为百分数，IWSLT 为 SacreBLEU；ARC-C 用 acc_norm，ARC-E 用 acc，与原 Table 2 口径对齐。', '',
          '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
report += ['| ' + ' | '.join(display(row[c]) for c in columns) + ' |' for row in rows]
report += ['', '## 缺项与使用说明', '',
           '- 强化 HAWQ 的 INT4/FP4 分配点目前有 NLL 与速度，缺完整下游任务，不能拿旧 HAWQ 策略的任务分数代填。',
           '- HAWQ minimum-cost control 的 P+D 与 run001 已评测 phase-uniform 策略 map 完全相同，生成质量继承该点；该控制点不能用来证明 Hessian 分配优于统一分配。Prefill 策略不同，精度仍待补。',
           '- HAWQ 行的 q 是构造其延迟上限时配对的 DASO 预算标签，不是 HAWQ 自身满足的质量预算；还存在 minimum-cost clamp，不能据此声称两方法同一质量约束。',
           '- 最新 run011 没有记录 P+D 首 token 时刻，不能从总时长推造 TTFT/TPOT，也不能把独立 Prefill-only 的 TTFT 填到不同 P+D 策略上。BF16 的两个 1.00 仅为归一化恒等值。',
           '- 若保留全部旧 uniform 行，还需在同一测速协议补齐这些行；原 run002 的对应精度和速度记录均仍保留。',
           '- DASO + 8bit 实际为原五候选加 SQ-INT8、FP8，共七候选；文件名 daso8 是历史 ID。原 DP 代表点与 exact/validation 改进版分别列出。',
           '- exact/validation epsilon=.01 在 Prefill 选中 SQ-INT8，在 P+D 回退 BF16；epsilon=.04 两侧均为异构策略。二者均在完整下游评测前固定，不叫 Max-speed/Balanced 新最优点。',
           '- run010 报告中的 ARC-E 为 normalized accuracy；此表读取原始 acc，避免混用 acc 与 acc_norm。',
           '- run011 的首轮/最后一轮存在排名变化；此表记录其最终轮并在 provenance.json 保留各已用点首轮加速比。run012 已做局部重测，不能将此表单轮差距写成稳定显著领先。', '',
           '## 精确缺项', '']
report += [f'- {r["method"]}: ' + ', '.join(r['missing']) for r in provenance if r['missing']]
report += ['', '[机器可读表格](main_table.csv) · [逐项来源与哈希](provenance.json) · [LaTeX](main_table.tex) · [生成代码](build_table.py)', '']
(HERE / 'README.md').write_text('\n'.join(report))
(HERE / 'provenance.json').write_text(json.dumps({'status': 'incomplete', 'speed_campaign': '011 quiet',
    'sources_sha256': sources, 'rows': provenance}, indent=2, ensure_ascii=False) + '\n')
tex = ['% Incomplete evidence table; -- means not measured under the selected protocol.',
       r'\begin{tabular}{llrrrrrrrrrrr}', r'\hline', ' & '.join(headers) + r' \\', r'\hline']
for row in rows:
    tex.append(' & '.join((display(row[c]).replace('待补', '--').replace('_', r'\_')) for c in columns) + r' \\')
tex += [r'\hline', r'\end{tabular}', '']
(HERE / 'main_table.tex').write_text('\n'.join(tex))
print(json.dumps({'rows': len(rows), 'filled_metric_cells': sum(row[c] != '' for row in rows for c in columns[2:]),
                  'missing_metric_cells': sum(len(r['missing']) for r in provenance), 'source_files': len(sources)}))
