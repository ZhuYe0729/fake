"""Append two exploratory P+D points and their references; retain all old rows."""
import csv
import hashlib
import io
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

DEST = Path(__file__).resolve().parent
REPO = DEST.parents[4]
RUNS = REPO / 'artifacts/rebuttal/runs'
BASE = DEST / 'reference/before_run032_033'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: list(csv.DictReader(p.open()))


def main():
    if not BASE.exists():
        assert len(load(DEST / 'main_table.csv')) == 19
        BASE.mkdir(parents=True)
        for name in ['README.md', 'main_table.csv', 'provenance.json']:
            shutil.copy2(DEST / name, BASE / name)
        (BASE / 'snapshot_sha256.json').write_text(json.dumps(
            {p.name:sha(p) for p in BASE.iterdir()}, indent=2) + '\n')
    for name, digest in read(BASE / 'snapshot_sha256.json').items():
        assert sha(BASE / name) == digest, name
    old = load(BASE / 'main_table.csv')
    assert len(old) == 19
    fields = list(old[0])
    old_by_name = {r['method']:r for r in old}
    retained_sq = 'DASO + 8bit + exact/validation, epsilon=.01'
    retained_i8 = 'DASO-8 incl. plain INT8 + exact/validation, epsilon=0.01'
    source = REPO / 'artifacts/rebuttal/responses/reviewer2/qwen3_budget_prefill_weight'
    source_provenance = read(source / 'provenance.json')
    data = {r['label']:r for r in load(source / 'complete_comparison.csv')}
    sources = read(BASE / 'provenance.json')['sources_sha256'].copy()

    def register(path):
        path = path.resolve()
        relative = str(path.relative_to(REPO))
        sources[relative] = sha(path)
        return relative

    for name in ['main_table.csv', 'README.md', 'provenance.json', 'snapshot_sha256.json']:
        register(BASE / name)
    register(Path(__file__))
    register(source / 'complete_comparison.csv')
    register(source / 'provenance.json')
    for rid in ['032', '033']:
        root = RUNS / rid
        audit = read(root / 'results/final_audit.json')
        assert audit['complete'] and audit['verified']
        assert sha(root / 'results/final_audit.json') == source_provenance[rid]['audit_sha256']
        for name, digest in audit['csv_sha256'].items():
            assert sha(root / 'results' / name) == digest
            register(root / 'results' / name)
        for name in ['results/final_audit.json', 'configs/generation_jobs.json',
                     'configs/formal_rounds.json', 'configs/design.json']:
            register(root / name)

    def speed(rid, pid):
        root = RUNS / rid
        result = {}
        records = []
        for i in range(3):
            paths = [root / f'results/measurements/formal_round{i}_{p}.json' for p in
                [pid, 'uniform_dense_bf16', 'uniform_dense_bf16_bracket_end', 'uniform_int8']]
            values = [read(p) for p in paths]
            for p, value in zip(paths, values):
                register(p)
                assert value['placement']['cpu_affinity'] == list(range(8))
                assert sum(not s['warmup'] for s in value['cases'][0]['samples']) == 5
                assert all(a['exact_equal'] for a in value['exact_weight_audit'].values())
            current, first, last, int8 = [v['cases'][0]['medians'] for v in values]
            records.append({'round':i, **current, 'e2e_vs_int8':int8['e2e_ms']/current['e2e_ms'],
                **{key+'_vs_bf16':statistics.mean([first[key],last[key]])/current[key]
                    for key in ['ttft_ms','tpot_ms','e2e_ms']}})
        for metric, key in [('pd_ttft_speedup','ttft_ms_vs_bf16'),
                            ('pd_tpot_speedup','tpot_ms_vs_bf16'),('pd_e2e_speedup','e2e_ms_vs_bf16')]:
            ratios = [r[key] for r in records]
            result[metric] = {'value':statistics.mean(ratios), 'rounds':ratios,
                             'min':min(ratios), 'max':max(ratios)}
        return result, records

    specs = [('uniform_dense_bf16','032','BF16 (032 reference)','BF16'),
        ('uniform_int8','032','INT8 W8A8 (032 reference)',retained_i8),
        ('uniform_sq_int8','032','SmoothQuant INT8 (032 reference)','SmoothQuant INT8'),
        ('w1_e0.01','032','DASO-8 exploratory P+D, q=.01, wP=1 (032)',retained_sq),
        ('w32_e0.03','033','DASO-8 exploratory P+D, q=.03, wP=32 (033)',retained_sq)]
    rows, origins, details, round_rows = [], [], [], []
    pre_keys = ['arc_c','arc_e','wino','mmlu','prefill_avg_acc','prefill_ttft_speedup']
    for label, rid, method, pre_source in specs:
        d = data[label]
        assert d['campaign'] == rid
        pid, root = d['policy'], RUNS / rid
        speedups, rounds = speed(rid, pid)
        assert abs(speedups['pd_e2e_speedup']['value']-float(d['speedup_bf16'])) < 1e-12
        assert abs(statistics.mean(r['e2e_vs_int8'] for r in rounds)-float(d['speedup_int8'])) < 1e-12
        jobs = read(root / 'configs/generation_jobs.json')
        generation_sources = {}
        for dataset, key, metric, samples in [('cnn_dm_1000','cnn_dm','rougeL_percent',1000),
                ('dsum','dialogsum','rougeL_percent',1500),('IWSLT','iwslt','sacre_bleu',333)]:
            reuse = next((j for j in jobs['reused'] if (j['policy'],j['dataset'])==(pid,dataset)),None)
            path = Path(reuse['source']) if reuse else root / f'results/generation/{pid}/{dataset}/metrics_0000_end.json'
            value = read(path)
            assert value['samples'] == samples and abs(value[metric]-float(d[key])) < 1e-12
            if reuse:
                assert sha(path) == reuse['sha256']
            generation_sources[key] = {'source':register(path),'metric':metric,'samples':samples}
        pre = old_by_name[pre_source]
        row = {'model':'Qwen3-8B','method':method, **{k:pre[k] for k in pre_keys},
               **{k:float(d[k]) for k in ['cnn_dm','dialogsum','iwslt']},
               'pd_avg_score':statistics.mean(float(d[k]) for k in ['cnn_dm','dialogsum','iwslt']),
               **{k:v['value'] for k,v in speedups.items()}}
        assert abs(float(row['prefill_avg_acc'])-statistics.mean(float(row[k]) for k in pre_keys[:4])) < 1e-10
        assert abs(row['pd_avg_score']-float(d['generation_avg'])) < 1e-12
        rows.append(row)
        for record in rounds:
            round_rows.append({'method':method,'run':rid,'policy':pid,**record})
        policy_path = root / f'configs/policies/{pid}.json'
        register(policy_path)
        mapping = read(policy_path)['method_map']
        counts = {p:dict(Counter(v[p+'_method'] for v in mapping.values())) for p in ['prefill','decode']}
        assert all(sum(v.values())==144 for v in counts.values())
        pre_policy = 'uniform_dense_bf16' if pre_source=='BF16' else 'uniform_int8' if pre_source==retained_i8 else 'uniform_sq_int8'
        origin = {'method':method,'run':rid,'policies':{'prefill_only':pre_policy,'prefill_only_reused_row':pre_source,
            'prefill_decode':pid}, 'prefill_only_source':str((BASE/'main_table.csv').relative_to(REPO)),
            'prefill_only_reused_fields':pre_keys,'prefill_only_newly_optimized':False,
            'prefill_decode_quality_sources':generation_sources,'speed_ranges':speedups,
            'prefill_decode_counts':counts,
            'pd_own_prefill_mc_avg':float(d['mc_avg']), 'proxy_budget':float(d['budget']) if d['budget'] else None,
            'prefill_weight':float(d['prefill_weight']) if d['prefill_weight'] else None, 'delta_nll':float(d['delta_nll']),
            'guarded_delta_nll':float(d['guarded_delta_nll']),
            'guard_within_nominal_budget':d['guard_within_nominal_budget']=='True' if d['guard_within_nominal_budget'] else None,
            'validation_selected_final_point':False,'same_campaign_e2e_vs_int8':float(d['speedup_int8'])}
        origins.append(origin)
        details.append({'method':method,'run':rid,'policy':pid,'prefill_only_policy':pre_policy,
            'prefill_only_reused_row':pre_source,
            **{k:d[k] for k in ['budget','prefill_weight','weighted_prediction','original_prediction',
                'delta_nll','guarded_delta_nll','mean_within_nominal_budget','guard_within_nominal_budget',
                'mc_avg','generation_avg','ttft_ms','tpot_ms','e2e_ms','speedup_int8',
                'speedup_int8_min','speedup_int8_max','prefill_sq_int8','prefill_int8',
                'prefill_dense_fp4','prefill_sparse_fp4','decode_w4','decode_sq_int8']},
            'prefill_method_counts':json.dumps(counts['prefill'],sort_keys=True),
            'decode_method_counts':json.dumps(counts['decode'],sort_keys=True)})

    def csv_text(items, columns):
        handle = io.StringIO(newline='')
        writer = csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        writer.writerows(items)
        return handle.getvalue()

    before = (BASE / 'main_table.csv').read_bytes()
    appended = csv_text(rows,fields).split('\r\n',1)[1].encode()
    assert (DEST / 'main_table.csv').read_bytes() in [before,before+appended], 'Unexpected edits to current table'
    (DEST / 'main_table.csv').write_bytes(before+appended)
    (DEST / 'new_point_details.csv').write_bytes(csv_text(details,list(details[0])).encode())
    (DEST / 'new_speed_rounds.csv').write_bytes(csv_text(round_rows,list(round_rows[0])).encode())
    old_md = (BASE / 'README.md').read_text()
    old_lines = [s for s in old_md.splitlines() if s.startswith('| ')][2:]
    assert len(old_lines) == 19
    new_lines = ['| '+r['method']+' | '+' | '.join(f'{float(r[k]):.2f}' for k in fields[2:])+' |' for r in rows]
    md = old_md.replace(old_lines[-1],old_lines[-1]+'\n'+'\n'.join(new_lines),1)
    md = md.replace(md.splitlines()[0],'# Qwen3-8B完整Table 2：保留原19行＋5行新参考/探索点（R1/R2/R3/R4/R5）',1)
    old_intro = md.split('\n\n')[1]
    md = md.replace(old_intro,'**24行、312项指标完整（含48项派生均值）。** 原19行全部数值保留：前17行来自015，后两行来自016。本次在其后追加032的BF16/INT8/SQ-INT8参考，以及032/033的两个P+D探索点。全部为分批实验汇总，速度按各自三轮BF16首尾锚点归一化；原19行文件快照见reference/before_run032_033，原始run和17行封存表不改。',1)
    md = md.replace('；三轮3warmup+8timed。','；原19行三轮3warmup+8timed，新增行P+D三轮3warmup+5timed。新增Prefill-only沿用原已测配置及其协议。',1)
    md = md.replace('最后两行的Prefill和P+D分别选择各自配置。','原run016两行的Prefill和P+D分别选择各自配置。',1)
    md += '\n## 本次新增行的口径\n\n新增DASO行的q和wP只决定P+D；Prefill-only沿用旧`DASO + 8bit + exact/validation, epsilon=.01`的全SQ-INT8配置，四项精度和独立TTFT均逐值复用，不称为本轮重新优化。新增INT8参考的Prefill-only沿用016全INT8结果；BF16/SQ参考沿用其已有独立结果。各来源见provenance.json。\n\nP+D配置自身的MC只作诊断，未填入主表Prefill-only列。所有新增行的完整生成均已测完，复用的同map/源码/输入协议数据带SHA记录。两项探索点来自已观测benchmark上的参数探索，不标为经过validation筛选的严格ε合格点，也不宣称通用框架优化或每项任务全胜。\n\n原INT8行与新增INT8参考来自不同实验记录，分数并不相同；本轮相对优势应对照新增参考及各自同轮INT8，不将历史分数混成同一次实验。033的普通INT8同轮E2E为1225.51ms；032为1220.93ms。\n\n| 新P+D点 | 生成Avg | E2E ms | ×同轮INT8 | P+D自身prefill MC Avg | 实测ΔNLL | 保护值 | 名义代理预算 | 保护值通过 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n'
    md = md.replace('033的普通INT8同轮E2E为1225.51ms；032为1220.93ms。',
        '033的普通INT8同轮E2E为1225.51ms；032为1220.93ms。新增BF16参考取每轮开头的记录、分母取首尾BF16平均，所以个别速度列舍入后为1.01。',1)
    for d in details[3:]:
        md += '| '+d['policy']+' | '+' | '.join(f'{float(d[k]):.5f}' for k in
            ['generation_avg','e2e_ms','speedup_int8','mc_avg','delta_nll','guarded_delta_nll','budget'])+' | '+d['guard_within_nominal_budget']+' |\n'
    md += '\n`.01/wP=1`：P=82SQ＋26INT8＋33稠密FP4＋3稀疏FP4，D=122W4A16＋22SQ；`.03/wP=32`：P=92SQ＋19INT8＋33稠密FP4、零稀疏，D=144W4A16。均为完整144模块×64动作自由求解结果。\n\n实际NLL使用既有WikiText test固定100窗口，保护值=mean+2SE+.005。两个重点点均未通过对应名义预算，保留失败标记；加权预算是偏好得分，不等同原始ΔNLL保证。速度均为单批P+D，恢复和下一批cycle不计入主指标。三轮范围见new_speed_rounds.csv，033的进程间资源等待见原run报告，不计入E2E。\n\n[新增行详情](new_point_details.csv) · [新行三轮速度](new_speed_rounds.csv) · [全部8标签及对照、失败点与曲线](../qwen3_budget_prefill_weight/README.md) · [032报告](../../../runs/032/REPORT.md) · [033报告](../../../runs/033/REPORT.md) · [重建脚本](append_run032_033.py)\n'
    assert all(line in md for line in old_lines)
    (DEST / 'README.md').write_text(md)
    meta = read(BASE / 'provenance.json')
    meta.update({'rows':24,'metric_cells':312,'original_metric_cells':264,'derived_metric_cells':48,
        'retained_rows':19,'retained_snapshot':str(BASE.relative_to(REPO)),
        'appended_run032_033':origins,'sources_sha256':sources,
        'scope':'Original19 rows retained; 3 references and 2 exploratory P+D points appended. Prefill-only reused independently, not replaced by P+D MC. No claim of passed nominal NLL budgets.'})
    meta['delivery_sha256'] = {n:sha(DEST/n) for n in ['README.md','main_table.csv','new_point_details.csv','new_speed_rounds.csv']}
    (DEST / 'provenance.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    final = load(DEST / 'main_table.csv')
    assert len(final)==24 and final[:19]==old and (DEST/'main_table.csv').read_bytes().startswith(before)
    for path,digest in sources.items():
        assert sha(REPO/path)==digest,path
    print('Verified24 rows: original19 byte prefix unchanged; 5 new rows,65 metrics, sources and three-round speeds checked.')


if __name__ == '__main__':
    main()
