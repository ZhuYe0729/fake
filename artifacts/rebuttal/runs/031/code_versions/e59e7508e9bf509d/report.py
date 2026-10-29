"""Audit the prospective full-space comparison, including failed proposals."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from common import RUN, write_json
from selection import guard

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, rows):
    with (RUN / 'results' / name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    design = read(RUN / 'configs/design.json')
    selected = read(RUN / 'configs/selected_policies.json')
    for path, digest in design['source_sha256'].items():
        assert sha(Path(path)) == digest, path
    assert selected['design_sha256'] == sha(RUN / 'configs/design.json')
    assert selected['pool_sha256'] == sha(RUN / 'configs/proposal_pool.json')
    assert selected['registry_sha256'] == sha(RUN / 'configs/policy_registry.json')
    assert selected['validation_sha256'] == sha(RUN / 'results/validation_candidates.json')
    for pid, digest in selected['calibration_sha256'].items():
        assert sha(RUN / f'results/measurements/calibration_{pid}.json') == digest
    for pid, row in read(RUN / 'configs/policy_registry.json').items():
        assert sha(RUN / f'configs/policies/{pid}.json') == row['policy_sha256']
    validation = read(RUN / 'results/validation_candidates.json')
    save('validation.csv', validation)
    baseline = read(RUN / 'results/confirmation/uniform_dense_bf16/prefill_decode.json')
    confirm = {}
    for path in (RUN / 'results/confirmation').glob('*/*.json'):
        value = read(path)
        assert len(value['blocks']) == 64 and value['data_role'] == 'confirmation'
        assert value['data_sha256'] == design['datasets']['confirmation']['sha256']
        assert path.stat().st_mtime >= selected['frozen_at_unix']
        delta = [a['avg_nll'] - b['avg_nll'] for a, b in zip(value['blocks'], baseline['blocks'], strict=True)]
        confirm[path.parent.name] = guard(delta, path.parent.name == 'uniform_dense_bf16')
    save('confirmation.csv', [{'policy': p, **v} for p, v in confirm.items()])
    campaigns, speed = [], {}
    rounds = read(RUN / 'configs/formal_rounds.json')
    assert rounds['selected_sha256'] == sha(RUN / 'configs/selected_policies.json')
    for i, order in enumerate(rounds['rounds']):
        records = {pid: read(RUN / f'results/measurements/formal_round{i}_{pid}.json') for pid in order}
        anchor = statistics.mean(records[p]['cases'][0]['medians']['e2e_ms'] for p in [order[0], order[-1]])
        int8 = records['uniform_int8']['cases'][0]['medians']['e2e_ms']
        for pid, record in records.items():
            assert record['input_sha256'] == design['datasets']['speed_formal']['sha256']
            assert record['placement']['cpu_affinity'] == list(range(8))
            assert all(v['exact_equal'] for v in record['exact_weight_audit'].values())
            assert sum(not s['warmup'] for s in record['cases'][0]['samples']) == 5
            if pid.endswith('_bracket_end'):
                continue
            row = {'policy': pid, 'round': i, **record['cases'][0]['medians'],
                   'speedup_bf16': anchor / record['cases'][0]['medians']['e2e_ms'],
                   'speedup_int8': int8 / record['cases'][0]['medians']['e2e_ms']}
            campaigns.append(row)
            speed.setdefault(pid, []).append(row)
    save('formal_campaigns.csv', campaigns)
    speed = {p: {key: statistics.mean(r[key] for r in rows) for key in rows[0] if key not in ['policy', 'round']}
             for p, rows in speed.items()}
    save('formal_speed.csv', [{'policy': p, **v} for p, v in speed.items()])
    rows = []
    for choice in selected['selected']:
        pid, budget = choice['policy'], choice['quality_budget']
        rows.append({'model': choice['model'], 'budget': budget, 'policy': pid,
            'validation_guard': choice['validation_and_calibration']['guarded_delta_nll'],
            'confirmation_mean': confirm[pid]['delta_nll'], 'confirmation_guard': confirm[pid]['guarded_delta_nll'],
            'mean_pass': confirm[pid]['delta_nll'] <= budget, 'guard_pass': confirm[pid]['guarded_delta_nll'] <= budget,
            **speed[pid]})
    save('selected_results.csv', rows)
    counts = []
    for pid in read(RUN / 'configs/policy_registry.json'):
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        counts.append({'policy': pid, **{phase: dict(Counter(v[phase + '_method'] for v in mapping.values()))
                                        for phase in ['prefill', 'decode']}})
    write_json(RUN / 'results/policy_counts.json', counts)
    fit = read(RUN / 'results/model_fit.json')
    lines = ['# Qwen3：完整候选空间质量修正快速实验', '',
        '144个模块独立选择prefill/decode，8×8动作；分桶仅是共享校准特征，不约束模块同选。旧/新使用相同LUT及精确MILP，允许signed质量成本。两者均以BF16预测为0，旧bias移除的影响单独记录。', '',
        f'复用训练期标签：{fit["train_policies"]}个train、{fit["holdout_policies"]}个policy holdout，前64个相同训练窗口。016 eager与029 Graph来源以各自BF16配对；BF16最大逐窗口差{fit["bf16_protocol_max_difference"]:.3g}，不宣称所有混合策略协议严格等价。', '',
        f'旧锚定代理holdout MAE={fit["old_holdout_mae"]:.6f}，signed残差={fit["new_holdout_mae"]:.6f}；ridge={fit["chosen_regularization"]}由固定policy holdout选择，这不是最终新窗口误差。', '',
        '128新validation决定质量可行性，同卡calibration E2E决定选点；冻结后64新confirmation。所有新质量/速度输入排除原训练及028/029/030窗口。保护值同mean+2SE+.005。主目标是原论文单批E2E，包含prefill→decode实际切换，批间恢复单列，不强加到主指标。', '',
        '| 模型 | ε | 策略 | Val guard | Confirm mean | Confirm guard | Guard pass | E2E ms | ×BF16 | ×INT8 |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["budget"]} | {r["policy"]} | {r["validation_guard"]:.5f} | '
            f'{r["confirmation_mean"]:.5f} | {r["confirmation_guard"]:.5f} | {r["guard_pass"]} | '
            f'{r["e2e_ms"]:.2f} | {r["speedup_bf16"]:.3f} | {r["speedup_int8"]:.3f} |')
    lines += ['', '各模型只使用自身四目标提案和共同anchor。030的90-W4仅是共同incumbent，不能将复用它算作本轮完整空间新发现。所有失败提案保留；不按确认或正式速度重选。本轮没有完整生成/MC，NLL可行不等于所有下游任务不退化，也未新增HAWQ对照。', '',
        '3轮GPU0/NUMA0，3warmup+5timed、首尾BF16、native decode Graph及6GiB KV。原始轮次与恢复开销保留。', '',
        '[所有validation](results/validation.csv) · [选择与确认](results/selected_results.csv) · [三轮速度](results/formal_campaigns.csv) · [各策略精度分配](results/policy_counts.json) · [模型拟合](results/model_fit.json)', '']
    (RUN / 'REPORT.md').write_text('\n'.join(lines))
    assert not list((RUN / 'data/checkpoints').rglob('*.safetensors'))
    evidence = [*list((RUN/'results/validation').glob('*/*.json')), *list((RUN/'results/confirmation').glob('*/*.json')),
                *list((RUN/'results/measurements').glob('*.json'))]
    write_json(RUN / 'results/final_audit.json', {'complete': True, 'verified': True,
        'code_sha256': {p.name: sha(p) for p in (RUN / 'code').glob('*.py')},
        'evidence_sha256': {str(p.relative_to(RUN)): sha(p) for p in evidence},
        'csv_sha256': {p.name: sha(p) for p in (RUN / 'results').glob('*.csv')},
        'source_files_verified': len(design['source_sha256']), 'new_calibration_gpu_jobs': 0,
        'selection_before_confirmation': True, 'primary_guard_pass': all(r['guard_pass'] for r in rows if r['model']=='signed_residual')})
    print('Audited full-space rapid study:', rows, flush=True)


if __name__ == '__main__':
    main()
