"""Build the reviewable report only from the fully validated final measurements."""
import csv
import datetime
import json
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def read(relative):return json.loads((RUN/relative).read_text())


def number(row,key,places=3):
    value=row.get(key,'')
    return f'{float(value):.{places}f}' if value!='' else '—'


def label(row):
    policy=row['policy_id']
    names={'uniform_dense_bf16':'BF16','uniform_dense_nvfp4':'Dense NVFP4',
        'uniform_sparse_bf16':'Sparse BF16','uniform_sparse_nvfp4':'Sparse NVFP4',
        'uniform_w4a16_ours':'NVFP4 W4A16','uniform_int8':'INT8 W8A8',
        'uniform_marlin_int4':'INT4 W4A16','phase_uniform_nvfp4_marlin':'阶段统一 hybrid'}
    name=names.get(policy)
    if name is None:
        name=f'DASO ε={row["quality_budget"]}' if row['family']=='DASO' else ('HAWQ FP4 ' if 'control' in row['family'] else 'HAWQ INT4 ')+policy.rsplit('_',1)[-1]
    return f'[{name}](configs/policies/{policy}.json)'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def preparation():
    result=[]
    for title,path,key in [('SparseGPT BF16','results/canonical_sparse_bf16.json','elapsed_seconds'),
        ('SparseGPT NVFP4','results/canonical_sparse_nvfp4.json','elapsed_seconds'),
        ('HAWQ Hessian','results/hessian_trace_blockwise_p16_b4_s256.json','elapsed_seconds'),
        ('HAWQ INT4 分配','results/hawq_marlin_int4_allocation.json','solve_seconds'),
        ('HAWQ FP4 分配','results/hawq_fp4_allocation.json','solve_seconds'),
        ('局部输出误差','results/local_error_metadata.json','elapsed_seconds')]:
        value=read(path)
        if key in value:result.append({'stage':title,'seconds':value[key],'source':path})
    for scenario in ['prefill_only','prefill_decode']:
        commands=list((RUN/'logs').glob(f'cal_{scenario}_p??_*.command.json'))
        start=min(path.stat().st_mtime for path in commands)
        end=(RUN/f'logs/daso_calibration_{"prefill" if scenario=="prefill_only" else "decode"}_01.log').stat().st_mtime
        result.append({'stage':f'DASO {scenario} 标签采集壁钟','seconds':end-start,
            'start_utc':datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),
            'end_utc':datetime.datetime.fromtimestamp(end,datetime.timezone.utc).isoformat(),
            'parallelism':'1 GPU' if scenario=='prefill_only' else 'first 12 labels serial, remaining labels on GPUs 0/1/2',
            'source':f'logs/cal_{scenario}_*.command.json and outer completion log'})
        costs=read(f'results/daso/{scenario}/fit_search_cost.json')
        result.extend([{'stage':f'DASO {scenario} 质量拟合','seconds':costs['quality_fit_seconds']},
                       {'stage':f'DASO {scenario} 延迟查询与 DP','seconds':costs['latency_query_and_dp_seconds']}])
    (RUN/'results/preparation_costs.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


def main():
    assert read('results/progress.json')['complete']
    with (RUN/'results/summary_final.csv').open() as handle:rows=list(csv.DictReader(handle))
    assert all(row['speed_source']=='common_gpu3' for row in rows)
    full=[row for row in rows if row['full_downstream']=='True']
    pre=[row for row in full if row['scenario']=='prefill_only']
    dec=[row for row in full if row['scenario']=='prefill_decode']
    costs=preparation()
    lines=['# Qwen3-8B 补充实验结果','',
        f'已完成：{len(set(row["policy_id"] for row in rows))} 份策略配置（按 policy_id 计）、{len(rows)} 个场景/NLL 点；'
        f'{len(pre)} 个 prefill 配置全量评测五项任务，{len(dec)} 个 decode 配置各评测 CNN/DM 1000、DialogSum 1500、IWSLT 333 条样本。',
        '', '正式速度和内存峰值均在同一张 RTX 5090（物理 GPU 3）测量，B=8、S=2048、O=1/64，固定 3 GiB BF16 KV、2 次 warmup + 5 次测量。'
        '生成采用原始样本与任务指令、Qwen3 non-thinking 模板、greedy、最多 256 新 token、固定 6 GiB KV。',
        '', '完整机器可读数据：[summary_final.csv](results/summary_final.csv)。'
        '协议和格式适配见 [README](README.md) 与 [final_protocol.json](configs/final_protocol.json)。',
        '', '## 实测质量预算','',
        '预算是质量模型预测约束，不能写成实际质量保证。表中违约量为 max(0, 实测 ΔNLL − ε)。'
        'Prefill NLL 计分 204700 个 token，decode NLL 计分 6400 个 token；均使用独立 WikiText 测试文本。','']
    quality=[row for row in rows if row['family']=='DASO']
    lines.append(table(['场景','ε','预测 ΔNLL','测试 ΔNLL','违约量','实测加速'],
        [['P' if row['scenario']=='prefill_only' else 'P+D',row['quality_budget'],number(row,'predicted_delta_nll',5),
          number(row,'measured_delta_nll',5),number(row,'budget_violation',5),
          number(row,'ttft_speedup' if row['scenario']=='prefill_only' else 'e2e_speedup')+'×'] for row in quality]))
    for scenario in ['prefill_only','prefill_decode']:
        group=[row for row in quality if row['scenario']==scenario]
        metrics=read(f'results/daso/{scenario}/quality_metrics.json')
        values=read(f'results/daso/{scenario}/quality_predictions.json')
        labels=[row['actual_delta_nll'] for row in values]
        violations=[float(row['budget_violation']) for row in group]
        lines+=['',f'{scenario}：{sum(value>1e-8 for value in violations)}/{len(group)} 点发生实际违约，最大 {max(violations):.5f} nats/token；'
            f'54 个训练策略 MAE={metrics["train"]["mae"]:.5f}，18 个留出策略 MAE={metrics["holdout"]["mae"]:.5f}，'
            f'校准标签 ΔNLL 范围 [{min(labels):.5f}, {max(labels):.5f}]。']
    zero=read('results/daso/prefill_only/solver_quality_integration_audit.json')[0]
    lines+=['','留出划分针对策略，仍在同一批训练文本上采集标签；上表的测试文本独立。'
        f'Prefill ε=0 的策略含 {sum(zero["zero_predicted_cost_compressed_actions"].values())} 个零预测代价压缩动作。'
        '质量成本重算与搜索导出值误差低于 1e-12，未发现成本对接不一致。',
        '', '对零预算点及三个预先选定代表点，额外测量了原校准文本的真实 NLL。下表分解选中策略的校准预测误差和校准→测试差值；这些诊断没有用于重新拟合或选择策略。','']
    audit=read('results/selected_policy_quality_audit.json')['rows']
    lines.append(table(['策略','预测 ΔNLL','校准实测 ΔNLL','测试实测 ΔNLL','校准预测误差','校准→测试差值'],
        [[row['policy_id'],*[f'{row[key]:.5f}' for key in ['predicted_delta_nll','measured_calibration_delta_nll',
            'measured_test_delta_nll','prediction_error_on_calibration','calibration_to_test_shift']]] for row in audit]))
    lines+=['', '![质量拟合与预算核验](results/figures/quality_calibration_and_budgets.png)',
        '', '## 完整下游结果','', '准确率、ROUGE-L 为百分数，BLEU 为 SacreBLEU 分数；PPL 为 word-PPL。','']
    lines.append(table(['配置','TTFT ms','加速','测试 ΔNLL','PPL','Wino','ARC-E','ARC-C norm','MMLU'],
        [[label(row),number(row,'ttft_ms',1),number(row,'ttft_speedup')+'×',number(row,'measured_delta_nll',5),
          *[number(row,key,2) for key in ['wikitext_word_ppl','winogrande_acc','arc_easy_acc','arc_challenge_acc_norm','mmlu_acc']]] for row in pre]))
    lines+=['',table(['配置','E2E ms','加速','TPOT ms','测试 ΔNLL','CNN/DM R-L','DialogSum R-L','IWSLT BLEU'],
        [[label(row),number(row,'e2e_ms',1),number(row,'e2e_speedup')+'×',number(row,'tpot_ms',2),number(row,'measured_delta_nll',5),
          *[number(row,key,2) for key in ['cnn_dm_rougeL','dialogsum_rougeL','iwslt_sacrebleu']]] for row in dec]),
        '', '![实际质量与延迟](results/figures/measured_quality_latency.png)',
        '', '低损失区间放大图如下；完整视图保留所有实测点。线段仅用于连接同族非支配实测点。',
        '', '![低损失区间](results/figures/measured_quality_latency_zoom.png)',
        '', '## 内存和准备成本','',
        '下表为程序周期的 PyTorch allocator 峰值（含初始化、固定 KV、工作区），CPU 权重缓存是在模型加载后按实际 tensor 字节计数。'
        'RSS 包含进程及映射文件，不等于常驻权重缓存；序列化权重大小另见 CSV 和 tensor_footprint.json。','']
    memory=[row for row in dec if row['family']=='DASO' or row['policy_id'] in
            ['uniform_dense_bf16','uniform_dense_nvfp4','uniform_marlin_int4','phase_uniform_nvfp4_marlin']]
    lines.append(table(['配置','GPU peak GiB','GPU reserved GiB','CPU prefill cache GiB','CPU decode cache GiB','pinned cache GiB','RSS peak GiB'],
        [[label(row),*[number(row,key,2) for key in ['peak_allocated_gib','peak_reserved_gib','host_prefill_weight_cache_gib',
            'host_decode_weight_cache_gib','pinned_host_weight_cache_gib','peak_host_rss_gib']]] for row in memory]))
    lines+=['',table(['准备阶段','壁钟秒'],[[row['stage'],f'{row["seconds"]:.2f}'] for row in costs]),
        '', '标签采集包含模型导出/加载等开销；decode 首 12 个标签串行，剩余标签分到三张卡。'
        '第 11 个策略父进程日志耗时包含等待并行区间的时间，不能将各日志时长直接相加。各准备阶段存在重叠，不应把上表全部相加作为总壁钟。',
        '', '## 复现口径与边界','',
        '- HAWQ-v3 是 Hessian 敏感度 + ILP 分配器的 Qwen3 PTQ 移植，未复现原 CNN QAT/TVM integer-only 流水线。'
        '主候选 INT4 W4A16 / INT8 W8A8 / BF16；另有 NVFP4 / INT8 / BF16 控制。BF16 是新增锚点，QKV、gate/up 融合决策绑定，prefill/decode 方法绑定。'
        '252 个块中 35 个负 trace 均值保留，未静默截断。来源：[官方 HAWQ](https://github.com/Zhen-Dong/HAWQ)，'
        '[固定 revision 与哈希](reference/sources.json)。',
        '- INT4 使用原生 BF16 输入/输出的 Marlin group-128 kernel。INT8 使用真实 torch._int_mm，包含激活量化、padding 和缩放；'
        '安装版 vLLM CUTLASS INT8 拒绝 SM120。该 INT8 基线没有 SmoothQuant smoothing。所有格式保留 embedding、LM head、norm 的 BF16 权重。',
        '- Sparse BF16 使用 2:4；Sparse NVFP4 使用后端所需的成对 4:8 权重模式。所有 kernel 均真实执行。'
        'HAWQ 延迟限制与 DASO 质量预算单位不同，比较必须依据实际 NLL/任务指标和延迟。',
        '- 速度仅覆盖本轮固定 B/S/O，模型为 Qwen3-8B；这些数据不能支持 30B、MoE 或混合在线调度的结论。'
        '代码、输入、所有成功和失败日志/结果均保留；仅任务自建的可重建临时模型二进制按清理记录释放。',
        '', '代码：[code](code/)；冻结输入：[data](data/)；完整性与 SHA-256：[manifest.json](results/manifest.json)。'
        '统一速度测量位于 results/speed_common_gpu3，原始重复测量、显存和 CPU 缓存元数据均可逐行追溯。','']
    (RUN/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
