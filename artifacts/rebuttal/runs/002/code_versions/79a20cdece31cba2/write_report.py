import csv
import json
from collections import Counter
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def load(path):
    with path.open() as f:return list(csv.DictReader(f))


def number(row,key,digits=3):
    value=row.get(key,'')
    return f'{float(value):.{digits}f}' if value!='' else '—'


def best(rows,scenario,limit):
    metric='ttft_speedup' if scenario=='prefill_only' else 'e2e_speedup'
    valid=[r for r in rows if r['scenario']==scenario and r.get('measured_delta_nll') and
        float(r['measured_delta_nll'])<=limit and r.get(metric)]
    if not valid:return '—'
    row=max(valid,key=lambda r:float(r[metric]))
    return f'{float(row[metric]):.3f}× ({row["policy_id"]})'


def main():
    rows=load(RUN/'results/summary_final.csv')
    points=json.loads((RUN/'configs/evaluation_points.json').read_text())['points']
    lines=['# Qwen3-8B：真实 8bit baseline 与扩展 DASO',
        '', '本轮新增 SmoothQuant INT8、原生 FP8 和 SmoothQuant-FP8 对照；DASO 在原五候选上加入 SmoothQuant INT8 与原生 FP8。',
        '所有新策略均先冻结再测独立测试集。新增候选组与五候选控制组使用完全相同的质量拟合和模块实测延迟表。',
        '', '## 完整 baseline', '',
        '| 方法 | Prefill ΔNLL | TTFT ms | Prefill 加速 | Decode ΔNLL | E2E ms | E2E 加速 | GPU allocated GiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['family']!='uniform' or row['scenario']!='prefill_only':continue
        other=next(r for r in rows if r['policy_id']==row['policy_id'] and r['scenario']=='prefill_decode')
        lines.append('| '+ ' | '.join([row['policy_id'],number(row,'measured_delta_nll',6),number(row,'ttft_ms',1),
            number(row,'ttft_speedup')+'×',number(other,'measured_delta_nll',6),number(row,'e2e_ms',1),
            number(row,'e2e_speedup')+'×',number(row,'peak_allocated_gib',2)])+' |')
    lines += ['', '旧七种 uniform baseline 的速度在本轮同卡重新测量；旧基线任务与质量证据复用 001 冻结结果，BF16 test NLL 重新测量校验。三种新增 8bit baseline 均完成完整任务。',
        '', '## 按实际质量比较候选扩展', '',
        '下表在实际测试 ΔNLL 不超过给定上限的已测点中选择最高加速。该表用于结果分析，不参与策略选择。', '',
        '| 场景 | 实际 ΔNLL 上限 | DASO + 8bit | 五候选，同质量拟合/延迟表 | 简单 phase-uniform |',
        '| --- | ---: | --- | --- | --- |']
    groups={family:[r for r in rows if r['family']==family] for family in ['daso8','daso5_measured','phase-uniform']}
    for scenario in ['prefill_only','prefill_decode']:
        for limit in [.005,.01,.02,.04,.07,.1,.15,.22,.45]:
            lines.append('| '+' | '.join([scenario,str(limit),*[best(groups[f],scenario,limit) for f in groups]])+' |')
    lines += ['', '![实测质量与延迟](results/figures/quality_latency_zoom.png)',
        '', '001 的 HAWQ 与原 DASO 在图中只作为历史参考：本轮 INT8 激活量化与输出缩放使用融合实现，候选集合与成本输入也已变化，不能据此把跨轮差异归因于分配算法。',
        '', '## 预算满足情况', '', '| 候选组 | 场景 | 测试点 | 超预算点 | 最大超预算 nats/token |',
        '| --- | --- | ---: | ---: | ---: |']
    for family in ['daso8','daso5_measured']:
        for scenario in ['prefill_only','prefill_decode']:
            values=[r for r in rows if r['family']==family and r['scenario']==scenario]
            violations=[float(r['budget_violation']) for r in values]
            lines.append(f'| {family} | {scenario} | {len(values)} | {sum(v>1e-9 for v in violations)} | {max(violations,default=0):.6f} |')
    lines += ['', '随机 policy holdout 误差不能作为搜索后策略的预算保证；不使用测试结果重新拟合、回选或校正本轮策略。',
        '', '## 求解离散化诊断', '',
        '原 2000-bin 全局网格在小误差 8bit 动作上可能过粗。Uniform SmoothQuant-INT8 的 prefill 原始预测 ΔNLL 为 0.001799，离散后为 0.028430；它的原始预测 Linear 成本 443.16 ms，低于预算 0.01 时 DP 选中策略的 497.92 ms，但离散可行域排除了该 uniform 配置。',
        '这与质量代理的预测误差是两个问题。本轮仍使用原求解器和原网格；此处只审计已有训练标签、拟合与离散成本，不修改策略或超参数。详见 [uniform_grid_audit.json](results/uniform_grid_audit.json)。',
        '', '## 下游任务', '', '| 方法/策略 | WikiText PPL | WinoGrande % | ARC-Easy % | ARC-Challenge % | MMLU % | CNN/DM ROUGE-L | DialogSum ROUGE-L | IWSLT BLEU |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['full_downstream']!='True':continue
        keys=['wikitext_word_ppl','winogrande_acc','arc_easy_acc','arc_challenge_acc_norm','mmlu_acc',
              'cnn_dm_rougeL','dialogsum_rougeL','iwslt_sacrebleu']
        lines.append('| '+' | '.join([row['policy_id']+' / '+row['scenario'],*[number(row,k,2) for k in keys]])+' |')
    usage=[]
    for p in points:
        policy=json.loads((RUN/f'configs/policies/{p["policy_id"]}.json').read_text())
        counts={phase:dict(Counter(v[phase+'_method'] for v in policy['method_map'].values())) for phase in ['prefill','decode']}
        usage.append({**p,'candidate_module_counts':counts})
    (RUN/'results/candidate_usage.json').write_text(json.dumps(usage,indent=2)+'\n')
    stratified={}
    for family in ['daso8','daso5_measured']:
        for scenario in ['prefill_only','prefill_decode']:
            predictions=json.loads((RUN/f'results/{family}/{scenario}/quality_predictions.json').read_text())
            groups={}
            for split in ['train','holdout']:
                for source in ['old','new']:
                    values=[abs(p['actual_delta_nll']-p['predicted_delta_nll']) for p in predictions
                        if p['split']==split and p['policy_id'].startswith('cal8_')==(source=='new')]
                    groups[source+'_'+split]={'count':len(values),'mae':sum(values)/len(values)}
            stratified[family+'/'+scenario]=groups
    (RUN/'results/quality_metrics_stratified.json').write_text(json.dumps(stratified,indent=2)+'\n')
    supplemental=json.loads((RUN/'configs/supplemental_fastest.json').read_text())
    lines += ['', '除原先冻结的预算代表点外，追加了每组速度 sweep 中实测最快且尚未覆盖的点的完整任务。选择仅使用 TTFT/E2E，不使用 NLL 或下游分数；本轮唯一追加点为 '+', '.join(p['policy_id'] for p in supplemental['additional'])+'。原预算代表点全部保留。',
        '', '## 实现与协议',
        '', '- INT8：真实 `torch._int_mm`，INT8×INT8→INT32，融合 Triton BF16 epilogue；M<32 补到 32 后丢弃多余输出。',
        '- FP8：真实 vLLM SM120 CUTLASS E4M3 W8A8 GEMM，BF16 输出。动态 per-token 激活、per-output-channel 权重；注意力和 KV 保持 BF16。',
        '- 旧 sparse BF16 使用 2:4；sparse NVFP4 使用后端要求的 paired 4:8。W4A16 NVFP4 与 native INT4 group128 Marlin 分别标记，不混用。',
        '- SmoothQuant：采用[官方实现](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py)的 alpha=0.5 公式与 Llama-like 范围，Qwen3 QKV 和 gate/up 平滑，o/down 不平滑。训练集 128×2048 tokens 收集通道最大值。',
        '- 为支持不同模块/阶段独立候选，x/s 融入候选内部激活量化，而非改动共享 RMSNorm；所有实际开销计入计时。平滑 FP8 明确标记为扩展对照。',
        '- 初始 128 项 shape/kernel 数值与有限值检查通过；最终搜索延迟来自 GPU3 的 64 项完整 forward、多轮同步 wall-time 测量。NVFP4 包装层不支持 CUDA Graph，失败尝试与原始单次计时保留。',
        '- 两组搜索共用 90 training / 30 policy holdout 标签，其中每场景 72 个原标签复用、48 个新增标签；质量模型、正则、4 个深度桶与 2000-bin 保守 DP 沿用原实现。',
        '- 模块延迟包含平滑、激活量化、GEMM、输出缩放与 Python dispatch；完整模型速度额外包含阶段切换、权重传输、注意力等全部开销。',
        '- 正式速度同一物理 GPU3、B8/S2048/O1 或 O64、2 warmups + 5 timed iterations、固定 3GiB BF16 KV；生成固定 6GiB KV、原冻结输入、greedy 256 tokens。',
        '- 完整 likelihood 任务使用 001 已归档 Arrow 样本，直接本地加载；61 个任务数据配置已核验。原 Hub 离线失败日志保留。',
        '- CPU 双表示缓存、GPU allocator peak、reserved peak 和主机 RSS 均保留在原始 metadata / CSV。导出体积与运行时内存分别报告。',
        '', '完整策略及候选比例：[candidate_usage.json](results/candidate_usage.json)；完整数表：[summary_final.csv](results/summary_final.csv)；完整性审计：[final_audit.json](results/final_audit.json)。',
        '', '## 重现', '',
        '依赖与模型哈希见 configs；原模型、patched vLLM 和 CUTLASS wrapper 路径见 code/common.py，源 lineage 见 reference/sources.json。数据链接依赖同项目下冻结的 runs/001。',
        '先在临时目录准备两种 canonical SparseGPT state（可复用 001，否则用 prepare_sparse.py --method ... --output-root ... 重建），再执行 prepare_8bit.py、profile_stable.py、daso_pipeline.py prepare。',
        '并行启动 experiment_queue.py 的 cal0/cal1/cal2/primary 四个 role，分别绑定物理 GPU0/4/7/3；统一 OMP_NUM_THREADS=4、OPENBLAS_NUM_THREADS=1、HF_HUB_OFFLINE=1、HF_DATASETS_OFFLINE=1。队列已完成步骤自动跳过；失败日志不覆盖。',
        '质量阶段使用 quality_queue.py 的共享 policy claim，先补齐 NLL，再做完整任务；可在 GPU3 另启动 quality_queue.py --name phase_helper --wait-primary，等待全部统一测速完成后加入质量评测。',
        '速度 sweep 完成且 phase_helper 运行后，在 GPU3 执行 supplement_fastest.py，记录仅按实测速度选择的额外完整任务；脚本等待当前策略结束后补测，再恢复共享队列。',
        '本次 residual_queue.py、switch_quality_queue.py 等脚本仅记录实际调度与恢复，干净复现使用标准 role 即可。临时模型二进制可以按归档 policy + activation_scales.pt + canonical state 重建；代码、日志、结果、文档全部保留。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
