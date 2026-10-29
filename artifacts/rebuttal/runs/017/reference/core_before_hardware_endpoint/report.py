"""Publish measured core rows and keep all other frozen points visibly pending."""
import argparse
import csv
import hashlib
import importlib.util
import math
import statistics

from common import REPO, RUN, SPEC, write_json
from launch import read


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN = RUN
    return module


def save(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized(pid, scene):
    output = 1 if scene == 'prefill_only' else 64
    fields = ['e2e_ms'] if output == 1 else ['ttft_ms','tpot_ms','e2e_ms']
    ratios = {key:[] for key in fields}
    for number in range(3):
        base = RUN / f'results/speed/node0/round{number}'
        value = read(base / f'{pid}.json')
        anchors = []
        for name in ['uniform_dense_bf16', 'uniform_dense_bf16_bracket_end']:
            anchors += [s for c in read(base / f'{name}.json')['cases'] if c['output'] == output
                        for s in c['samples'] if not s['warmup']]
        measured = [s for c in value['cases'] if c['output'] == output for s in c['samples'] if not s['warmup']]
        assert len(anchors)==16 and len(measured)==8
        for key in fields:
            ratios[key].append(statistics.median(s[key] for s in anchors)/
                statistics.median(s[key] for s in (anchors if pid=='uniform_dense_bf16' else measured)))
    return {key:statistics.median(values) for key,values in ratios.items()}, ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    assert read(RUN / 'results/formal_complete.json')['complete']
    pool = read(RUN / 'configs/allocation_pool.json')
    selected = read(RUN / 'configs/selected_policies.json')
    assert selected['pool_sha256'] == hashlib.sha256((RUN / 'configs/allocation_pool.json').read_bytes()).hexdigest()
    primary = set(read(RUN / 'configs/primary_policies.json')['policies'])
    analysis = load('common_matched_analysis', RUN.parent / '016/code/build_report.py')
    quality = load('frozen_complete_task_reader', RUN.parent / '013/code/write_report.py')
    points = []
    for point in pool['points']:
        pid, scene = point['policy_id'], point['scenario']
        row = {'label':point['label'], 'policy':pid, 'scenario':scene, 'family':point['family'],
            'latency_fraction':point.get('latency_fraction'), 'quality_budget':point.get('quality_budget'),
            'delta_nll':None, 'paired_se':None, 'e2e_ms':None, 'e2e_speedup':None,
            'cycle_ms':None, 'e2e_target_misses':None, 'status':'infeasible' if pid is None else 'speed_pending'}
        if pid:
            row['delta_nll'], row['paired_se'] = analysis.nll(pid, scene)
            if pid in primary:
                timing = analysis.speed(pid, scene)
                row.update(e2e_ms=timing['e2e_ms'],e2e_speedup=timing['speedup'],cycle_ms=timing['cycle_ms'],status='complete')
                target = point.get('actual_e2e_target_ms')
                if target:
                    row['e2e_target_misses'] = sum(s['e2e_ms']>target for s in timing['samples'])
        points.append(row)
    save(RUN / 'results/allocation_points.csv', points)
    lookup = {(p['label'],p['scenario']):p['policy_id'] for p in pool['points']}
    lookup.update({(p['label'],p['scenario']):p['policy'] for p in selected['selected']})
    labels = {'dense_bf16':'BF16','dense_nvfp4':'Dense NVFP4','w4a16_ours':'W4A16 NVFP4 Marlin',
        'sparse_bf16':'Sparse BF16 (SparseGPT)','sparse_nvfp4':'Sparse NVFP4 (SparseGPT)',
        'int8':'INT8 W8A8','sq_int8':'SmoothQuant INT8','fp8':'FP8 W8A8'}
    methods = [(labels[m], 'uniform_'+m, 'uniform_'+m) for m in [
        'dense_bf16','dense_nvfp4','w4a16_ours','sparse_bf16','sparse_nvfp4','int8','sq_int8','fp8']]
    methods.append(('Uniform SQ-INT8 prefill / W4A16 decode', 'uniform_sq_int8', 'phase_sq_int8_w4a16_ours'))
    methods += [(f'DASO-5 fixed original {kind}',f'original_daso5_prefill_only_{kind}',
                 f'original_daso5_prefill_decode_{kind}') for kind in ['balanced','max_speed']]
    methods.append(('HAWQ-FP4 PTQ r=.80','hawq_signed_prefill_only_same_operator_r0.8',
                    'hawq_signed_prefill_decode_phase_pair_r0.8'))
    methods += [(f'DASO-8 measured-LUT + exact/validation epsilon={e:g}',f'daso8_verified_prefill_only_q{e:g}',
                 f'daso8_verified_prefill_decode_q{e:g}') for e in [.01,.04]]
    rows, provenance, ranges = [], [], []
    for title, pp, pd in methods:
        row = {'model': 'Llama-3.1-8B-Instruct' if SPEC['run']=='017' else 'Llama-2-7B-Chat', 'method':title}
        trace = {'method':title,'policies':{},'cells':{}}
        for scene,label in [('prefill_only',pp),('prefill_decode',pd)]:
            pid = lookup[label,scene]
            assert pid and pid in primary
            trace['policies'][scene] = pid
            metrics = {'arc_c':None,'arc_e':None,'wino':None,'mmlu':None} if scene=='prefill_only' else {
                'cnn_dm':None,'dialogsum':None,'iwslt':None}
            if args.full:
                metrics, sources = quality.quality([SPEC['run'],pid],scene)
                trace['cells'].update(sources)
            row.update(metrics)
            ratios, repeated = normalized(pid,scene)
            if scene=='prefill_only':
                row['prefill_ttft_speedup'] = ratios['e2e_ms']
            else:
                row.update(pd_ttft_speedup=ratios['ttft_ms'],pd_tpot_speedup=ratios['tpot_ms'],pd_e2e_speedup=ratios['e2e_ms'])
            for metric, values in repeated.items():
                ranges.append({'method':title,'scenario':scene,'metric':metric,'median_speedup':statistics.median(values),
                    'min_speedup':min(values),'max_speedup':max(values)})
            delta,se = analysis.nll(pid,scene)
            trace[scene+'_confirmation'] = {'delta_nll':delta,'paired_se':se}
        rows.append(row)
        provenance.append(trace)
    save(RUN / 'results/main_table.csv', rows)
    save(RUN / 'results/speed_ranges.csv', ranges)
    write_json(RUN / 'results/provenance.json', provenance)
    if args.full:
        assert read(RUN / 'results/full_quality_complete.json')['complete']
        assert all(isinstance(v,(float,int)) and math.isfinite(v) for row in rows
            for key,v in row.items() if key not in ['model','method'])
        full = []
        for target in read(RUN / 'configs/full_quality_targets.json'):
            values, _sources = quality.quality([SPEC['run'],target['policy']],target['scenario'])
            for metric,value in values.items():
                full.append({**target,'metric':metric,'value':value})
        save(RUN / 'results/full_task_metrics.csv',full)
    columns = ['method','arc_c','arc_e','wino','mmlu','prefill_ttft_speedup','cnn_dm','dialogsum','iwslt',
               'pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']
    lines = [f'# {SPEC["model_id"]}：原模型强基线与DASO-8', '',
        '核心三轮速度与64窗口confirmation已完成。'+('预定完整任务也已完成。' if args.full else '完整任务仍在继续，待测单元明确留空。'), '',
        f'所有新速度统一RTX5090 GPU0/NUMA0、B8/S2048/O1或64、{SPEC["evaluation_kv_gib"]}GiB BF16 KV、按层arena复用及native decode Graph；'
        '三轮3warmup+8timed与BF16首尾锚点。原论文数值保留，以下是统一新路径重新评测。', '',
        'ARC-C acc_norm、ARC-E/Wino/MMLU acc；CNN/DM1000、DialogSum1500为ROUGE-L%，IWSLT333为SacreBLEU。'
        f'生成batch={SPEC["generation_batch"]}，使用原论文PMPD文字包装及模型默认BOS；总输入最多3840、greedy最多256新token允许EOS。完整生成全部本轮重评，固定O测速另行执行。', '',
        '| Method | ARC-C | ARC-E | Wino | MMLU | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D TTFT × | TPOT × | E2E × |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append('| '+' | '.join(str(row[c]) if c=='method' else ('待' if row[c] is None else f'{row[c]:.2f}') for c in columns)+' |')
    lines += ['', '## 比较范围与来源', '',
        '- HAWQ为Hessian/ILP的PTQ适配；FP4候选prefill用NVFP4、decode用W4A16，8bit两阶段均plain INT8。原目标不可行时明确记录，不静默放宽。',
        '- 原DASO-5 Balanced/Max speed保留原method-map；DASO-8新增目标模型误差/标签、精确加和代理求解与独立validation，不把所有收益归给原DP。',
        '- 原标签经过BF16/NVFP4采样协议检查；若失败，按预定规则重新计算全部原policy标签。数据来源和缓存决策见原始目录。',
        '- confirmation与validation分开；benchmark以前有整体评估，不称为全新语料。均值预算与paired误差不构成形式化质量保证。',
        '- 实测E2E不含请求间恢复，完整cycle及内存/初始化成本另存；三轮范围不是置信区间。',
        '- 全部冻结点均有NLL；非核心速度控制若尚未测，CSV标为speed_pending，不能用于实测排名。先完成两模型核心主表再扩展。', '',
        '[主表CSV](main_table.csv) · [全部点](allocation_points.csv) · [逐项来源](provenance.json) · [三轮范围](speed_ranges.csv)', '']
    text='\n'.join(lines)
    (RUN / 'REPORT.md').write_text(text.replace('](main_table.csv)','](results/main_table.csv)').replace('](allocation_points.csv)','](results/allocation_points.csv)').replace('](provenance.json)','](results/provenance.json)').replace('](speed_ranges.csv)','](results/speed_ranges.csv)'))
    output = REPO / f'artifacts/rebuttal/responses/reviewer2/{SPEC["model_id"]}_table2'
    output.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ['main_table.csv','allocation_points.csv','provenance.json','speed_ranges.csv']:
        shutil.copy2(RUN / 'results' / name, output / name)
    if args.full:
        shutil.copy2(RUN / 'results/full_task_metrics.csv', output / 'full_task_metrics.csv')
    (output / 'README.md').write_text(text+f'\n[原始代码、数据、日志与配置](../../../runs/{SPEC["run"]}/README.md)\n')
    write_json(RUN / 'results/report_complete.json',{'complete':True,'full_tasks_complete':args.full,
        'main_rows':len(rows),'allocated_points':len(points),'primary_policies':len(primary),
        'supplemental_speed_pending':sum(p['status']=='speed_pending' for p in points)})
    print('Published core table',SPEC['model_id'],'rows',len(rows),'full tasks',args.full,flush=True)


if __name__ == '__main__':
    main()
